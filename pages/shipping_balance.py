from dash import html, dcc, dash_table, callback, Output, Input, State, Dash, ALL
from dash.dash_table.Format import Format, Group, Scheme
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import datetime as dt
import json
from io import StringIO
from dash.exceptions import PreventUpdate
import configparser
import os
from sqlalchemy import create_engine

from fundamentals.kpler_fundamentals import *

############################################ postgres sql connection ###################################################
#------ code to be able to access config.ini, even having the path in the .virtualenvs is not working without it ------#
try:
    # Get the directory where your script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate to the directory containing config.ini
    # Adjust the number of '..' as needed to reach the correct directory
    config_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))  # Go up one level
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





def prepare_table_data(df, metric, selected_regions=None, selected_year=None, selected_statuses=None,
                       is_intracountry=False):
    """
    Prepare data for tables showing the metric values by region pair/country and vessel type.
    Args:
        df: DataFrame with trade data
        metric: The column to aggregate ('count_trades' or 'sum_ton_miles')
        selected_regions: List of origin shipping regions to filter by (only for non-intracountry)
        selected_year: Year to filter by (defaults to latest)
        selected_statuses: List of status values to filter by
        is_intracountry: Whether this is for intracountry data
    Returns:
        DataFrame formatted for display in a DataTable
    """
    # Filter the data for years 2019 and later
    filtered_df = df[df['year'] >= 2019]

    # Get the latest year if not specified
    if not selected_year or selected_year == "All Years":
        selected_year = filtered_df['year'].max()
    else:
        selected_year = int(selected_year)

    # Filter by year
    filtered_df = filtered_df[filtered_df['year'] == selected_year]

    # Apply filters specific to trade region or intracountry
    if is_intracountry:
        index_field = 'origin_country_name'
    else:
        # Create a new column combining origin and destination regions
        filtered_df['region_pair'] = filtered_df['origin_shipping_region'] + ' → ' + filtered_df[
            'destination_shipping_region']
        index_field = 'region_pair'

        # Apply region filter if selected (only for non-intracountry)
        if selected_regions and 'All Regions' not in selected_regions:
            filtered_df = filtered_df[filtered_df['origin_shipping_region'].isin(selected_regions)]

    # Apply status filter if selected
    if selected_statuses and 'All Statuses' not in selected_statuses:
        filtered_df = filtered_df[filtered_df['status'].isin(selected_statuses)]

    # Determine aggregation method based on metric type
    if metric.startswith('median_'):
        agg_method = 'median'
    elif metric.startswith('mean_'):
        agg_method = 'mean'
    elif metric in ['count_trades', 'sum_ton_miles']:
        agg_method = 'sum'
    else:
        # Default to mean for other metrics that might be averages
        agg_method = 'mean'
    
    # Aggregate the data
    agg_data = filtered_df.groupby([index_field, 'vessel_type'])[metric].agg(agg_method).reset_index()

    # Pivot the data for the table - vessel types as columns
    pivot_table = agg_data.pivot_table(
        index=[index_field],
        columns='vessel_type',
        values=metric,
        aggfunc=agg_method,
        fill_value=0
    ).reset_index()

    # Add a Total column
    vessel_cols = [col for col in pivot_table.columns if col != index_field]
    
    if agg_method in ['median', 'mean']:
        # For median/mean metrics, calculate overall median/mean across vessel types
        pivot_table['Total'] = pivot_table[vessel_cols].mean(axis=1)
    else:
        # For sum metrics, sum across vessel types
        pivot_table['Total'] = pivot_table[vessel_cols].sum(axis=1)

    # Format numeric values based on metric type
    if metric == 'count_trades':
        # Integer formatting for count data
        for col in vessel_cols + ['Total']:
            pivot_table[col] = pivot_table[col].astype(int)
    elif agg_method in ['median', 'mean']:
        # Round to 2 decimal places for median/mean metrics
        for col in vessel_cols + ['Total']:
            pivot_table[col] = pivot_table[col].round(2)

    # Add a Total row
    total_row = {index_field: 'Total'}
    for col in vessel_cols + ['Total']:
        if agg_method in ['median', 'mean']:
            # For median/mean metrics, calculate overall median/mean across regions
            value = pivot_table[pivot_table[index_field] != 'Total'][col].median() if agg_method == 'median' else pivot_table[pivot_table[index_field] != 'Total'][col].mean()
            total_row[col] = round(value, 2) if not pd.isna(value) else 0
        else:
            # For sum metrics, sum across regions
            total_row[col] = int(pivot_table[col].sum()) if metric == 'count_trades' else pivot_table[col].sum()
    pivot_table = pd.concat([pivot_table, pd.DataFrame([total_row])], ignore_index=True)

    # Sort by Total value in descending order (except Total row)
    non_total_rows = pivot_table[pivot_table[index_field] != 'Total'].copy()
    total_row = pivot_table[pivot_table[index_field] == 'Total'].copy()
    non_total_rows = non_total_rows.sort_values('Total', ascending=False)

    # Recombine sorted rows with total row at bottom
    pivot_table = pd.concat([non_total_rows, total_row], ignore_index=True)

    return pivot_table


def create_stacked_bar_chart(df, metric, title_suffix, selected_statuses=None, is_intracountry=False):
    """
    Create a Plotly visualization showing data by year, vessel type, and shipping regions/countries.
    Args:
        df: DataFrame with trade data
        metric: The column name to sum and visualize
        title_suffix: Text to use in the title describing the metric
        selected_statuses: List of status values to filter by
        is_intracountry: Whether this is for intracountry data
    Returns:
        A Plotly figure object
    """
    # Filter the data for years 2019 and later
    filtered_df = df[df['year'] >= 2019].copy()

    # Apply status filter if selected
    if selected_statuses and 'All Statuses' not in selected_statuses:
        filtered_df = filtered_df[filtered_df['status'].isin(selected_statuses)]

    # Set up color palette
    distinct_colors = [
        '#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4',
        '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990', '#dcbeff',
        '#9A6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1',
        '#000075', '#a9a9a9', '#008080', '#e6beff', '#9a6324', '#fffac8',
        '#800000', '#aaffc3', '#808000', '#ffd8b1', '#e6194b', '#3cb44b'
    ]

    # Set grouping field based on data type
    if is_intracountry:
        group_field = 'origin_country_name'
        chart_title = f'Intracountry {title_suffix} by Year, Vessel Type, and Origin Country (2019+)'
        legend_title = 'Origin Country'
    else:
        # Create a new column combining origin and destination regions
        filtered_df['region_pair'] = filtered_df['origin_shipping_region'] + ' → ' + filtered_df[
            'destination_shipping_region']
        group_field = 'region_pair'
        chart_title = f'{title_suffix} by Year, Vessel Type, and Shipping Regions (2019+)'
        legend_title = 'Shipping Regions (Origin → Destination)'

    # Aggregate the filtered data
    stacked_data = filtered_df.groupby(['year', 'vessel_type', group_field])[metric].sum().reset_index()

    # Get unique values
    years = sorted(stacked_data['year'].unique())
    vessel_types = sorted(stacked_data['vessel_type'].unique())
    group_values = sorted(stacked_data[group_field].unique())

    # Create figure
    fig = go.Figure()

    # Create a dictionary to keep track of the legend items added
    legend_items = set()

    # Calculate position for each year-vessel combination
    group_width = 0.8  # Width for all bars in a year-vessel group
    year_spacing = 0.2  # Space between different years
    bar_width = group_width / len(vessel_types)

    # Calculate positions for the x-axis
    x_positions = {}
    for y_idx, year in enumerate(years):
        for v_idx, vessel in enumerate(vessel_types):
            x_pos = y_idx * (1 + year_spacing) + v_idx * bar_width
            x_positions[(year, vessel)] = x_pos

    # Create a dictionary to store the cumulative stack values
    stack_values = {(year, vessel): 0 for year in years for vessel in vessel_types}

    # Create the stacked bars
    for i, group_value in enumerate(group_values):
        color = distinct_colors[i % len(distinct_colors)]

        for year in years:
            for vessel in vessel_types:
                # Filter data for this combination
                data = stacked_data[(stacked_data['year'] == year) &
                                    (stacked_data['vessel_type'] == vessel) &
                                    (stacked_data[group_field] == group_value)]

                if not data.empty:
                    value = data[metric].values[0]
                    x_pos = x_positions[(year, vessel)]

                    # Add a trace for this segment of the stacked bar
                    showlegend = group_value not in legend_items
                    if showlegend:
                        legend_items.add(group_value)

                    fig.add_trace(go.Bar(
                        x=[x_pos],
                        y=[value],
                        name=group_value,
                        marker_color=color,
                        showlegend=showlegend,
                        base=stack_values[(year, vessel)],
                        width=bar_width * 0.95  # Make bars slightly narrower than the allocated space
                    ))

                    # Update the stack value for the next segment
                    stack_values[(year, vessel)] += value

    # Create custom x-axis ticks and labels
    x_ticks = []
    x_labels = []

    # Add year labels at the center of each year group
    for y_idx, year in enumerate(years):
        year_center = y_idx * (1 + year_spacing) + (len(vessel_types) * bar_width) / 2 - bar_width / 2
        x_ticks.append(year_center)
        x_labels.append(str(year))

    # Add vessel type abbreviations below each bar
    vessel_annotations = []
    for y_idx, year in enumerate(years):
        for v_idx, vessel in enumerate(vessel_types):
            x_pos = x_positions[(year, vessel)]

            # Extract first 2 letters of vessel type for the annotation
            vessel_abbr = vessel[:2] if len(vessel) >= 2 else vessel

            vessel_annotations.append(
                dict(
                    x=x_pos,
                    y=0,
                    text=vessel_abbr,
                    showarrow=False,
                    xanchor='center',
                    yanchor='top',
                    yshift=-20,  # Position below x-axis
                    font=dict(size=8, color='#6b7280'),  # Small, subtle text
                    textangle=0,  # Keep horizontal for readability
                    xref='x',
                    yref='y'
                )
            )

    # Add a table-like annotation for vessel type abbreviations
    vessel_abbrs = [vessel[:2] if len(vessel) >= 2 else vessel for vessel in vessel_types]
    vessel_info = '<br>'.join([f"{abbr}: {full}" for abbr, full in zip(vessel_abbrs, vessel_types)])
    vessel_legend = dict(
        x=0.01,
        y=0,
        text=f"<b>Vessel Types:</b><br>{vessel_info}",
        showarrow=False,
        xanchor='left',
        yanchor='bottom',
        bordercolor='black',
        borderwidth=1,
        bgcolor='white',
        font=dict(size=10),
        xshift=10,
        yshift=10,
        align='left'
    )

    # Update layout with professional styling from dash_style.md
    fig.update_layout(
        
        # X-Axis Professional Styling
        xaxis=dict(
            title=dict(text='Year', font=dict(size=13, color='#374151')),
            tickvals=x_ticks,
            ticktext=x_labels,
            tickangle=0,
            tickmode='array',  # Use array mode to ensure custom ticks are used
            showgrid=True,
            gridcolor='rgba(200, 200, 200, 0.3)',  # Subtle grid
            gridwidth=0.5,
            linecolor='#d1d5db',  # Subtle gray borders
            linewidth=1,
            tickfont=dict(size=11, color='#6b7280')
        ),
        
        # Y-Axis Professional Styling
        yaxis=dict(
            title=dict(text=title_suffix, font=dict(size=13, color='#374151')),
            showgrid=True,
            gridcolor='rgba(200, 200, 200, 0.3)',
            gridwidth=0.5,
            linecolor='#d1d5db',
            linewidth=1,
            tickfont=dict(size=11, color='#6b7280'),
            zeroline=True,
            zerolinecolor='rgba(150, 150, 150, 0.4)',
            zerolinewidth=1
        ),
        
        barmode='stack',
        
        # Professional Legend Positioning (keep on the right as requested)
        legend=dict(
            x=1.02,
            y=1,
            xanchor='left',
            yanchor='top',
            title=dict(text=legend_title, font=dict(size=12, color='#374151')),
            bgcolor='rgba(255, 255, 255, 0)',  # Transparent
            bordercolor='rgba(255, 255, 255, 0)',
            borderwidth=0,
            font=dict(size=10, color='#374151'),
            itemsizing='constant',
            itemwidth=30
        ),
        
        annotations=vessel_annotations,
        
        # Professional Background and Margins
        plot_bgcolor='rgba(248, 249, 250, 0.5)',  # Subtle background
        paper_bgcolor='white',
        height=700,
        margin=dict(l=70, r=250, t=60, b=120),  # Increased bottom margin for vessel abbreviations
        
        # Enhanced Interactivity
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='rgba(255, 255, 255, 0.95)',
            bordercolor='rgba(200, 200, 200, 0.8)',
            font=dict(size=11, color='#1f2937'),
            align='left'
        ),
        
        # Smooth Animations
        transition=dict(duration=300, easing='cubic-in-out'),
        autosize=True
    )
    
    # Create vessel types footnote information
    vessel_abbrs = [vessel[:2] if len(vessel) >= 2 else vessel for vessel in vessel_types]
    vessel_footnote = ' | '.join([f"{abbr}: {full}" for abbr, full in zip(vessel_abbrs, vessel_types)])
    
    return fig, vessel_footnote


def global_shipping_balance(aggregation_level='monthly', life_expectancy=20, lng_view='demand', utilization_rate=0.85):
    """
        aggregation_level (str): Time aggregation level. Options:
            'monthly' - Year+Month (default)
            'quarterly' - Year+Quarter
            'seasonal' - Year+Season
            'yearly' - Year only
    """
    # ------------ step 1: read demand from Woodmack, using net imports which is conservative compared to gross imports ----#
    # Optimized single query with all transformations in SQL
    direction_mapping = {
        'demand': 'Import',
        'supply': 'Export'
    }
    
    if lng_view not in direction_mapping:
        raise ValueError(f"Invalid lng_view: {lng_view}. Must be 'demand' or 'supply'")
    
    direction = direction_mapping[lng_view]
    
    wm_query = '''
    WITH latest_short_term AS (
        SELECT 
            country_name,
            start_date::DATE as start_date,
            SUM(metric_value) / 12 * 1000 * 2222 AS value,
            publication_date,
            'Short Term' as source
        FROM at_lng.woodmac_gas_imports_exports_monthly__mmtpa
        WHERE publication_date = (
            SELECT MAX(publication_date) 
            FROM at_lng.woodmac_gas_imports_exports_monthly__mmtpa 
            WHERE release_type = 'Short Term Outlook'
        )
        AND release_type = 'Short Term Outlook'
        AND direction = %(direction)s
        AND metric_name = 'Flow'
        AND start_date<'2036-01-01'
        GROUP BY country_name, start_date, publication_date
        HAVING SUM(metric_value) > 0
    ),
    short_term_max_date AS (
        SELECT MAX(start_date) as max_date
        FROM latest_short_term
    ),
    latest_long_term AS (
        SELECT 
            country_name,
            start_date::DATE as start_date,
            SUM(metric_value) / 12 * 1000 * 2222 AS value,
            publication_date,
            'Long Term' as source
        FROM at_lng.woodmac_gas_imports_exports_monthly__mmtpa
        WHERE publication_date = (
            SELECT MAX(publication_date) 
            FROM at_lng.woodmac_gas_imports_exports_monthly__mmtpa 
            WHERE release_type = 'Long Term Outlook'
        )
        AND release_type = 'Long Term Outlook'
        AND direction = %(direction)s
        AND metric_name = 'Flow'
        AND start_date::DATE > (SELECT max_date FROM short_term_max_date)
        AND start_date<'2036-01-01'
        GROUP BY country_name, start_date, publication_date
        HAVING SUM(metric_value) > 0
    ),
    combined_data AS (
        SELECT * FROM latest_short_term
        UNION ALL
        SELECT * FROM latest_long_term
    ),
    enriched_data AS (
        SELECT 
            country_name AS "Country",
            start_date AS "Date",
            value,
            source,
            CASE 
                WHEN EXTRACT(MONTH FROM start_date) IN (10, 11, 12, 1, 2, 3) THEN 'W'
                ELSE 'S'
            END AS season,
            'Q' || EXTRACT(QUARTER FROM start_date)::TEXT AS quarter,
            EXTRACT(YEAR FROM start_date)::INTEGER AS year
        FROM combined_data
    )
    SELECT 
        ed."Country",
        ed."Date",
        ed.value,
        ed.season,
        ed.quarter,
        ed.year,
        ed.source,
        mc.shipping_region
    FROM enriched_data ed
    LEFT JOIN at_lng.mappings_country mc 
        ON ed."Country" = mc.country
    ORDER BY ed."Country", ed."Date"
    '''
    
    try:
        # Secure parameterized query execution
        wm_net_imports_mcm = pd.read_sql(wm_query, engine, params={'direction': direction},parse_dates='Date')
         
    except Exception as e:
        print(f"Error fetching WoodMac data: {str(e)}")
        # Return empty DataFrame on error to allow continuation
        wm_net_imports_mcm = pd.DataFrame()

    # ------------ step 2: kpler metrics by country --------------------------------------------#
    query_max_hist_date = '''
        select max("end") as hist_date_max
        from at_lng.kpler_trades
        where status='Delivered'
        and upload_timestamp_utc = (select max(upload_timestamp_utc) from at_lng.kpler_trades)
    '''
    hist_date_max = pd.read_sql(query_max_hist_date, engine)
    hist_date_max = pd.to_datetime(hist_date_max.hist_date_max.dt.date[0]).replace(day=1)

    _, df_trades_shipping_region = kpler_analysis(engine)

    # Define weighted mean function
    wm = lambda x, df, weight_col: np.round(np.average(x, weights=df.loc[x.index, weight_col]), 1)

    # Calculate metrics for current year (2024+)
    recent_df_laden = df_trades_shipping_region[(df_trades_shipping_region.status == 'laden') &
                                                (df_trades_shipping_region.year >= 2024)]
    recent_df_nonladen = df_trades_shipping_region[(df_trades_shipping_region.status == 'non_laden') &
                                                   (df_trades_shipping_region.year >= 2024)]

    # Calculate weighted metrics
    mean_shipping_days_laden = recent_df_laden.groupby(['destination_shipping_region', 'season', 'quarter']).agg(
        {'mean_delivery_days': lambda x: wm(x, recent_df_laden, "count_trades")}
    ).reset_index().rename(columns={'mean_delivery_days': 'mean_laden_days'})

    mean_shipping_days_nonladen = recent_df_nonladen.groupby(['origin_shipping_region', 'season', 'quarter']).agg(
        {'mean_delivery_days': lambda x: wm(x, recent_df_nonladen, "count_trades")}
    ).reset_index().rename(columns={'mean_delivery_days': 'mean_nonladen_days'})

    median_cargo_volumes = recent_df_laden.groupby(['destination_shipping_region', 'season', 'quarter']).agg(
        {'median_cargo_destination_cubic_meters': lambda x: wm(x, recent_df_laden, "count_trades")}
    ).reset_index()

    # Calculate metrics for all years
    # Generate calculations by year for historical data
    metrics_by_year = []
    for status, dest_field, field_name in [
        ('laden', 'destination_shipping_region', 'mean_laden_days'),
        ('non_laden', 'origin_shipping_region', 'mean_nonladen_days')
    ]:
        df_filtered = df_trades_shipping_region[df_trades_shipping_region.status == status]
        metric_calc = df_filtered.groupby([dest_field, 'season', 'year', 'quarter']).agg(
            {'mean_delivery_days': lambda x: wm(x, df_filtered, "count_trades")}
        ).reset_index()
        metric_calc = metric_calc.rename(columns={
            'mean_delivery_days': field_name,
            dest_field: 'shipping_region'
        })
        metrics_by_year.append(metric_calc)

    # Calculate cargo volumes by year
    cargo_volumes_by_year = df_trades_shipping_region[df_trades_shipping_region.status == 'laden'].groupby(
        ['destination_shipping_region', 'season', 'year', 'quarter']
    ).agg(
        {'median_cargo_destination_cubic_meters': lambda x: wm(x, df_trades_shipping_region[
            df_trades_shipping_region.status == 'laden'], "count_trades")}
    ).reset_index().rename(columns={'destination_shipping_region': 'shipping_region'})

    # Split data into historical and future
    wm_net_imports_mcm_hist = wm_net_imports_mcm[wm_net_imports_mcm.Date < hist_date_max]
    wm_net_imports_mcm_fut = wm_net_imports_mcm[wm_net_imports_mcm.Date >= hist_date_max]

    # Process historical data
    df_shipping_country_hist = wm_net_imports_mcm_hist.copy()
    for metric_df in metrics_by_year + [cargo_volumes_by_year]:
        df_shipping_country_hist = pd.merge(
            df_shipping_country_hist,
            metric_df,
            how='left',
            on=['shipping_region', 'season', 'year', 'quarter']
        )

    # Calculate ships demand for historical data
    df_shipping_country_hist['ships_demand'] = np.round(
        (df_shipping_country_hist['value'] / df_shipping_country_hist['median_cargo_destination_cubic_meters']) *
        (df_shipping_country_hist['mean_laden_days'] + df_shipping_country_hist['mean_nonladen_days']) /
        df_shipping_country_hist['Date'].dt.days_in_month, 1
    )

    df_shipping_country_hist = df_shipping_country_hist.dropna()

    # Calculate weighted aggregates for historical data
    wm_value = lambda x: np.round(np.average(x, weights=df_shipping_country_hist.loc[x.index, "value"]), 1)

    df_shipping_demand_hist = df_shipping_country_hist.groupby('Date').agg({
        'ships_demand': 'sum',
        'value': 'sum',
        'median_cargo_destination_cubic_meters': wm_value,
        'mean_laden_days': wm_value,
        'mean_nonladen_days': wm_value
    }).reset_index()

    # Process future data - similar structure as historical
    df_shipping_country_fut = pd.merge(
        wm_net_imports_mcm_fut,
        mean_shipping_days_laden.rename(columns={'destination_shipping_region': 'shipping_region'}),
        how='left',
        on=['shipping_region', 'season', 'quarter']
    )

    df_shipping_country_fut = pd.merge(
        df_shipping_country_fut,
        mean_shipping_days_nonladen.rename(columns={'origin_shipping_region': 'shipping_region'}),
        how='left',
        on=['shipping_region', 'season', 'quarter']
    )

    df_shipping_country_fut = pd.merge(
        df_shipping_country_fut,
        median_cargo_volumes.rename(columns={'destination_shipping_region': 'shipping_region'}),
        how='left',
        on=['shipping_region', 'season', 'quarter']
    )

    # Calculate ships demand for future data
    df_shipping_country_fut['ships_demand'] = np.round(
        (df_shipping_country_fut['value'] / df_shipping_country_fut['median_cargo_destination_cubic_meters']) *
        (df_shipping_country_fut['mean_laden_days'] + df_shipping_country_fut['mean_nonladen_days']) /
        df_shipping_country_fut['Date'].dt.days_in_month, 1
    )

    df_shipping_country_fut = df_shipping_country_fut.dropna()

    # Calculate weighted aggregates for future data
    wm_value = lambda x: np.round(np.average(x, weights=df_shipping_country_fut.loc[x.index, "value"]), 1)

    df_shipping_demand_fut = df_shipping_country_fut.groupby('Date').agg({
        'ships_demand': 'sum',
        'value': 'sum',
        'median_cargo_destination_cubic_meters': wm_value,
        'mean_laden_days': wm_value,
        'mean_nonladen_days': wm_value
    }).reset_index()

    # Combine historical and future data
    df_shipping_demand = pd.concat([df_shipping_demand_fut, df_shipping_demand_hist])

    # ------------ step 3: Import from syy new ships coming online -------------------------------------------------#
    # Get vessel data
    query_syy = '''
        select distinct "Name" as vessel_name,
            'In construction' as vessel_status,
            extract('year' from "Delivery") as vessel_build_year,
            "Delivery" as vessel_delivery_date,
            "Capacity" as vessel_capacity_cubic_meters,
            upload_timestamp_utc
        from at_lng.syy_newbuilds
        where upload_timestamp_utc = (select max(upload_timestamp_utc) from at_lng.syy_newbuilds)
    '''
    df_syy = pd.read_sql(query_syy, engine)
    
    query_kpler = '''
        select *
        from at_lng.kpler_vessels_info
        where (vessel_status ='Active' or (vessel_status ='Inactive' and last_port_call_end is not null))
            and vessel_build_year is not null
            and is_floating_storage=FALSE
            and upload_timestamp_utc = (select max(upload_timestamp_utc) from at_lng.kpler_vessels_info)
    '''
    df_kpler = pd.read_sql(query_kpler, engine)
    df_kpler['vessel_delivery_date'] = df_kpler.apply(lambda v: dt.datetime(int(v.vessel_build_year), 6, 1), axis=1)

    # Combine vessel data
    vessels_df = pd.concat([df_kpler, df_syy])

    # Calculate retirement dates
    vessels_df['theorical_retirement_date'] = pd.to_datetime(vessels_df.vessel_build_year + life_expectancy, format='%Y')
    vessels_df.loc[
        vessels_df.theorical_retirement_date < dt.datetime(2025, 2, 1), 'theorical_retirement_date'] = dt.datetime(2025,
                                                                                                                   2, 1)

    # Distribute retirement dates to avoid all vessels retiring at once  
    # Group and process each group separately to distribute retirement dates
    grouped = vessels_df.groupby('theorical_retirement_date', group_keys=False)
    dfs = []
    for retirement_date, group in grouped:
        n = len(group)
        months = np.linspace(0, 11, n, dtype=int)
        group = group.copy()
        group['theorical_retirement_date'] = [retirement_date + pd.DateOffset(months=int(m)) for m in months]
        dfs.append(group)
    
    # Combine all groups back together
    vessels_df = pd.concat(dfs, ignore_index=True) if dfs else vessels_df

    # Ensure timezone consistency before comparison
    # Handle mixed timezone-aware/naive timestamps in upload_timestamp_utc
    vessels_df['upload_timestamp_utc'] = pd.to_datetime(vessels_df['upload_timestamp_utc'], utc=True).dt.tz_localize(None)
    vessels_df['theorical_retirement_date'] = pd.to_datetime(vessels_df['theorical_retirement_date']).dt.tz_localize(None)
    
    # Calculate final retirement date
    vessels_df['retirement_date'] = np.where(
        vessels_df['vessel_status'] != 'Inactive',
        vessels_df[['upload_timestamp_utc', 'theorical_retirement_date']].max(axis=1),
        vessels_df['last_port_call_end']
    )

    # Normalize dates to first day of month
    vessels_df['retirement_date'] = vessels_df['retirement_date'].dt.to_period('M').dt.to_timestamp()
    vessels_df['vessel_delivery_date'] = vessels_df['vessel_delivery_date'].dt.to_period('M').dt.to_timestamp()

    # Calculate vessel statistics by month
    results = []
    for date_i in pd.date_range(start='1/1/2000', end='1/1/2040', freq='MS'):
        # Ships active previously and not removed in the month
        ships_before_month = vessels_df[(vessels_df['vessel_delivery_date'] < date_i) &
                                        (vessels_df['retirement_date'] > date_i)]

        # Ships delivered in the month
        ships_during_month = vessels_df[(vessels_df['vessel_delivery_date'] == date_i)]

        # Ships removed in the month
        ships_removed_during_month = vessels_df[(vessels_df['retirement_date'] == date_i)]

        # Calculate statistics
        total_active_ships = len(ships_before_month) + len(ships_during_month)

        if total_active_ships > 0:
            avg_ship_size = ((ships_before_month['vessel_capacity_cubic_meters'].sum() +
                              ships_during_month['vessel_capacity_cubic_meters'].sum()) /
                             total_active_ships).astype(int)
        else:
            avg_ship_size = 0

        results.append({
           'date': date_i,
            'total_active_ships': total_active_ships,
            'ships_added': len(ships_during_month),
            'ships_removed': len(ships_removed_during_month),
            'average_size_cubic_meters': avg_ship_size
        })

    # Create DataFrame with results
    ship_analysis_df = pd.DataFrame(results)

    # Merge shipping demand with vessel statistics
    df_global_shipping_balance = pd.merge(
        df_shipping_demand.rename(columns={'Date': 'date'}),
        ship_analysis_df,
        how='left',
        on='date'
    )

    # Calculate final metrics
    df_global_shipping_balance['ships_demand_total'] = np.round(
        (df_global_shipping_balance['value'] / (df_global_shipping_balance['average_size_cubic_meters'] * utilization_rate)) *
        (df_global_shipping_balance['mean_laden_days'] + df_global_shipping_balance['mean_nonladen_days']) /
        df_global_shipping_balance['date'].dt.days_in_month, 1
    )

    df_global_shipping_balance['net'] = df_global_shipping_balance['total_active_ships'] - df_global_shipping_balance[
        'ships_demand']
    df_global_shipping_balance['net_new'] = df_global_shipping_balance['total_active_ships'] - \
                                            df_global_shipping_balance['ships_demand_total']

    # Add additional time-based columns for aggregation
    df_global_shipping_balance['year'] = df_global_shipping_balance['date'].dt.year
    df_global_shipping_balance['quarter'] = df_global_shipping_balance['date'].dt.quarter
    df_global_shipping_balance['month'] = df_global_shipping_balance['date'].dt.month
    df_global_shipping_balance['season'] = np.where(
        df_global_shipping_balance['date'].dt.month.isin([10, 11, 12, 1, 2, 3]),
        'W', 'S'
    )

    # Apply time aggregation
    if aggregation_level == 'monthly':
        # No additional aggregation needed - already at month level
        return df_global_shipping_balance.sort_values('date')

    elif aggregation_level == 'quarterly':
        # Group by year and quarter
        agg_dict = {
            'total_active_ships': 'mean',
            'ships_demand': 'mean',
            'ships_demand_total': 'mean',
            'net': 'mean',
            'net_new': 'mean',
            'value': 'sum',
            'median_cargo_destination_cubic_meters': 'mean',
            'mean_laden_days': 'mean',
            'mean_nonladen_days': 'mean',
            'average_size_cubic_meters': 'mean',
            'ships_added': 'sum',
            'ships_removed': 'sum'
        }

        quarterly_df = df_global_shipping_balance.groupby(['year', 'quarter']).agg(agg_dict).reset_index()

        # Create a date column for the first day of each quarter
        quarterly_df['date'] = quarterly_df.apply(
            lambda row: pd.Timestamp(int(row['year']), int((row['quarter'] - 1) * 3 + 1), 1),
            axis=1
        )

        return quarterly_df.sort_values('date')

    elif aggregation_level == 'seasonal':
        # Group by year and season
        agg_dict = {
            'total_active_ships': 'mean',
            'ships_demand': 'mean',
            'ships_demand_total': 'mean',
            'net': 'mean',
            'net_new': 'mean',
            'value': 'sum',
            'median_cargo_destination_cubic_meters': 'mean',
            'mean_laden_days': 'mean',
            'mean_nonladen_days': 'mean',
            'average_size_cubic_meters': 'mean',
            'ships_added': 'sum',
            'ships_removed': 'sum'
        }

        seasonal_df = df_global_shipping_balance.groupby(['year', 'season']).agg(agg_dict).reset_index()

        # Create a date column - Winter (Jan 1) or Summer (Jul 1)
        seasonal_df['date'] = seasonal_df.apply(
            lambda row: pd.Timestamp(int(row['year']), 1 if row['season'] == 'W' else 7, 1),
            axis=1
        )

        return seasonal_df.sort_values('date')

    elif aggregation_level == 'yearly':
        # Group by year only
        agg_dict = {
            'total_active_ships': 'mean',
            'ships_demand': 'mean',
            'ships_demand_total': 'mean',
            'net': 'mean',
            'net_new': 'mean',
            'value': 'sum',
            'median_cargo_destination_cubic_meters': 'mean',
            'mean_laden_days': 'mean',
            'mean_nonladen_days': 'mean',
            'average_size_cubic_meters': 'mean',
            'ships_added': 'sum',
            'ships_removed': 'sum'
        }

        yearly_df = df_global_shipping_balance.groupby('year').agg(agg_dict).reset_index()

        # Create a date column for January 1st of each year
        yearly_df['date'] = yearly_df['year'].apply(lambda year: pd.Timestamp(int(year), 1, 1))

        return yearly_df.sort_values('date')

    else:
        # Default to monthly if invalid aggregation level
        return df_global_shipping_balance.sort_values('date')


def create_datatable(data, index_field):
    """Create a formatted DataTable from the provided data."""
    columns = []
    for col in data.columns:
        if col == index_field:
            display_name = "Origin Country" if index_field == "origin_country_name" else col
            columns.append({"name": display_name, "id": col})
        else:
            # Format numeric columns with thousand separators
            columns.append({
                "name": str(col),
                "id": str(col),
                "type": "numeric",
                "format": Format(
                    group=Group.yes,
                    scheme=Scheme.fixed,
                    precision=0,
                    group_delimiter=',',
                    decimal_delimiter='.'
                )
            })

    return dash_table.DataTable(
        columns=columns,
        data=data.to_dict('records'),
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'left',
            'padding': '5px',
            'minWidth': '100px',
            'maxWidth': '300px',
            'whiteSpace': 'normal',
        },
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold'
        },
        style_data_conditional=[
            {
                'if': {'row_index': len(data) - 1},  # Total row
                'fontWeight': 'bold',
                'backgroundColor': 'rgb(240, 240, 240)'
            }
        ],
        page_size=25,
        sort_action='native',
        filter_action='native',
        fill_width=False,
        export_format='xlsx',
        export_headers='display',
        export_columns='visible'
    )


# Dashboard layout
layout = html.Div([
    # Store components for caching data
    dcc.Store(id='trades-shipping-data-store', storage_type='local'),
    dcc.Store(id='shipping-balance-data-store', storage_type='local'),
    dcc.Store(id='shipping-balance-supply-data-store', storage_type='local'),
    dcc.Store(id='dropdown-options-store', storage_type='local'),
    dcc.Store(id='refresh-timestamp-store', storage_type='local'),
    dcc.Store(id='intracountry-data-store', storage_type='local'),

    # Global Shipping Balance Overview Section - Enterprise Standard with Inline Controls
    html.Div([
        # Enterprise Standard Inline Section Header with Controls
        html.Div([
            html.H3('Global Shipping Balance Overview', className="section-title-inline"),
            html.Label("Aggregation:", className="inline-filter-label"),
            dcc.Dropdown(
                id='aggregation-dropdown',
                options=[
                    {'label': 'Year+Quarter', 'value': 'quarterly'},
                    {'label': 'Year+Month', 'value': 'monthly'},
                    {'label': 'Year+Season', 'value': 'seasonal'},
                    {'label': 'Year', 'value': 'yearly'}
                ],
                value='quarterly',
                clearable=False,
                className='inline-dropdown'
            ),
            html.Label("Vessel displacement age:", className="inline-filter-label"),
            dcc.Input(
                id='vessel-age-input',
                type='number',
                value=20,
                min=1,
                max=50,
                step=1,
                className='filter-input',
                style={'width': '80px', 'height': '34px', 'fontSize': '14px', 'padding': '6px 8px'}
            ),
            html.Div(id='last-refresh-indicator', className='text-tertiary', style={'fontSize': '13px'})
        ], className="inline-section-header"),
        
        # Charts Container with Professional Layout
        html.Div([
            # Left column - Demand View
            html.Div([
                html.Div([
                    html.H4('Demand View', className="subheader-title-inline"),
                ], className="inline-subheader", style={'marginBottom': '12px'}),
                dcc.Graph(id='global-shipping-balance', style={'height': '400px'}),
            ], style={'flex': '1', 'paddingRight': '12px'}),
            
            # Right column - Supply View  
            html.Div([
                html.Div([
                    html.H4('Supply View', className="subheader-title-inline"),
                ], className="inline-subheader", style={'marginBottom': '12px'}),
                dcc.Graph(id='global-shipping-balance-supply', style={'height': '400px'}),
            ], style={'flex': '1', 'paddingLeft': '12px'}),
        ], style={'display': 'flex', 'gap': '24px'})
    ], className='section-container'),

    # Trade Analysis by Region Pairs Section - Enterprise Standard
    html.Div([
        # Enterprise Standard Inline Section Header with All Controls
        html.Div([
            html.H3('Trade Analysis by Region Pairs', className="section-title-inline"),
            html.Label("Select Metric:", className="inline-filter-label"),
            dcc.Dropdown(
                id='trade-metric-selector',
                options=[
                    {'label': 'Count Trades', 'value': 'count_trades'},
                    {'label': 'Ton Miles', 'value': 'ton_miles'},
                    {'label': 'Median Delivery Days', 'value': 'median_delivery_days'},
                    {'label': 'Median Mileage (Nautical Miles)', 'value': 'median_mileage_nautical_miles'},
                    {'label': 'Median Ton Miles', 'value': 'median_ton_miles'},
                    {'label': 'Median Utilization Rate', 'value': 'median_utilization_rate'},
                    {'label': 'Median Cargo Volume (m³)', 'value': 'median_cargo_destination_cubic_meters'},
                    {'label': 'Median Vessel Capacity (m³)', 'value': 'median_vessel_capacity_cubic_meters'}
                ],
                value='count_trades',
                clearable=False,
                className='inline-dropdown'
            ),
            html.Label("Origin Region:", className="inline-filter-label"),
            dcc.Dropdown(
                id='region-dropdown',
                options=[{'label': 'All Regions', 'value': 'All Regions'}],
                value=['All Regions'],
                multi=True,
                clearable=False,
                className='inline-dropdown-multi'
            ),
            html.Label("Year:", className="inline-filter-label"),
            dcc.Dropdown(
                id='year-dropdown',
                options=[{'label': 'All Years', 'value': 'All Years'}],
                value=None,
                clearable=False,
                className='inline-dropdown'
            ),
            html.Label("Status:", className="inline-filter-label"),
            dcc.Dropdown(
                id='region-status-dropdown',
                options=[{'label': 'All Statuses', 'value': 'All Statuses'}],
                value=['All Statuses'],
                multi=True,
                clearable=False,
                className='inline-dropdown-multi'
            ),
        ], className="inline-section-header"),
        
        # Vessel Types Information (between header and chart)
        html.Div([
            html.P([
                html.Span("Vessel Types: ", style={'fontWeight': '500', 'color': '#374151', 'fontSize': '13px'}),
                html.Span(id='vessel-types-info', style={'color': '#6b7280', 'fontSize': '13px'})
            ], style={
                'fontFamily': 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',
                'marginTop': '12px',
                'marginBottom': '16px',
                'paddingLeft': '16px',
                'paddingRight': '16px',
                'paddingTop': '10px',
                'paddingBottom': '10px',
                'backgroundColor': '#f8f9fa',
                'borderRadius': '4px',
                'border': '1px solid #e9ecef'
            })
        ]),
        
        # Single Chart Container
        html.Div([
            dcc.Graph(id='trade-analysis-visualization', style={'height': '600px'})
        ], style={'marginTop': '0px'}),
        
        # Single Table Container
        html.Div([
            html.Div(id='trade-analysis-table-container', style={'overflow-x': 'auto'})
        ], style={'marginTop': '24px'})
    ], className='section-container'),

    # Intracountry Trade Analysis Section
    html.Div([
        html.Div([
            html.H2("Intracountry Trade Analysis", className='section-title-inline'),
            html.P("Analysis of domestic shipping patterns by origin country", className='section-subtitle')
        ], className='header-content'),
        
        html.Div([
            html.Div([
                html.Label("Filter by Year:", className='filter-label'),
                dcc.Dropdown(
                    id='intracountry-year-dropdown',
                    options=[{'label': 'All Years', 'value': 'All Years'}],
                    value=None,
                    clearable=False,
                    className='inline-dropdown'
                )
            ], className='filter-group'),
            
            html.Div([
                html.Label("Filter by Status:", className='filter-label'),
                dcc.Dropdown(
                    id='intracountry-status-dropdown',
                    options=[{'label': 'All Statuses', 'value': 'All Statuses'}],
                    value=['All Statuses'],
                    multi=True,
                    clearable=False,
                    className='inline-dropdown'
                )
            ], className='filter-group')
        ], className='filter-bar')
    ], className='inline-section-header'),

    # Intracountry Trade Visualizations Section - Enterprise Standard
    html.Div([
        # Enterprise Standard Inline Section Header
        html.Div([
            html.H3('Intracountry Trade Visualizations', className="section-title-inline"),
        ], className="inline-section-header"),
        
        # Chart Container with Professional Layout
        html.Div([
            # Left column - Trade Count
            html.Div([
                html.Div([
                    html.H4('Count of Intracountry Trades', className="subheader-title-inline"),
                ], className="inline-subheader"),
                dcc.Graph(id='intracountry-count-visualization', style={'height': '600px'})
            ], style={'flex': '1', 'paddingRight': '12px'}),
            
            # Right column - Ton Miles
            html.Div([
                html.Div([
                    html.H4('Intracountry Ton Miles', className="subheader-title-inline"),
                ], className="inline-subheader"),
                dcc.Graph(id='intracountry-tonmiles-visualization', style={'height': '600px'})
            ], style={'flex': '1', 'paddingLeft': '12px'})
        ], style={'display': 'flex', 'gap': '24px', 'marginTop': '16px'})
    ], className='section-container'),

    # Intracountry Data Tables - Enterprise Standard
    html.Div([
        # Enterprise Standard Inline Section Header
        html.Div([
            html.H3('Intracountry Trade Data Tables', className="section-title-inline"),
        ], className="inline-section-header"),
        
        # Tables Container
        html.Div([
            html.Div([
                html.Div([
                    html.H4('Trade Count Data', className="subheader-title-inline"),
                ], className="inline-subheader"),
                html.Div(id='intracountry-count-table-container', style={'overflow-x': 'auto'})
            ], className='section-container', style={'flex': '1', 'marginRight': '12px'}),
            
            html.Div([
                html.Div([
                    html.H4('Ton Miles Data', className="subheader-title-inline"),
                ], className="inline-subheader"),
                html.Div(id='intracountry-tonmiles-table-container', style={'overflow-x': 'auto'})
            ], className='section-container', style={'flex': '1', 'marginLeft': '12px'})
        ], style={'display': 'flex', 'gap': '24px', 'marginTop': '16px'})
    ], style={'marginBottom': '32px'}),

])


# Callbacks
# Update the refresh_data callback to include the aggregation dropdown
@callback(
    Output('trades-shipping-data-store', 'data'),
    Output('shipping-balance-data-store', 'data'),
    Output('shipping-balance-supply-data-store', 'data'),
    Output('dropdown-options-store', 'data'),
    Output('refresh-timestamp-store', 'data'),
    Output('intracountry-data-store', 'data'),
    Input('global-refresh-button', 'n_clicks'),
    Input('aggregation-dropdown', 'value'),
    Input('vessel-age-input', 'value'),  # Add this input
    prevent_initial_call=False
)
def refresh_data(n_clicks, aggregation_level='monthly', vessel_age=20):
    """Fetch and prepare all data needed for the dashboard."""
    # Fetch shipping balance data with selected aggregation and vessel age
    df_global_shipping_balance = global_shipping_balance(aggregation_level, vessel_age)
    # Fetch shipping balance supply data (same parameters but with lng_view='supply')
    df_global_shipping_balance_supply = global_shipping_balance(aggregation_level, vessel_age, lng_view='supply')

    # Rest of the function remains the same
    # Fetch trade shipping data
    df_intracountry_trades, df_trades_shipping_region = kpler_analysis(engine)

    # Extract unique values for dropdown options
    # Filter for relevant years (2019+)
    df_filtered = df_trades_shipping_region[df_trades_shipping_region['year'] >= 2019]

    # Get origin regions for dropdown
    origin_regions = sorted(df_filtered['origin_shipping_region'].unique())
    region_options = [{'label': 'All Regions', 'value': 'All Regions'}] + [
        {'label': region, 'value': region} for region in origin_regions
    ]

    # Get years for dropdown
    years = sorted(df_filtered['year'].unique())
    latest_year = max(years)
    year_options = [{'label': 'All Years', 'value': 'All Years'}] + [
        {'label': str(year), 'value': str(year)} for year in years
    ]

    # Get status values for dropdowns
    region_statuses = sorted(df_trades_shipping_region['status'].unique())
    intracountry_statuses = sorted(df_intracountry_trades['status'].unique())

    status_options_region = [{'label': 'All Statuses', 'value': 'All Statuses'}] + [
        {'label': status.capitalize(), 'value': status} for status in region_statuses
    ]

    status_options_intracountry = [{'label': 'All Statuses', 'value': 'All Statuses'}] + [
        {'label': status.capitalize(), 'value': status} for status in intracountry_statuses
    ]

    # Create single-select version of status options for metrics dropdown
    status_options_single = [{'label': status['label'], 'value': status['value']}
                             for status in status_options_intracountry]

    # Store options data
    options_data = {
        'region_options': region_options,
        'year_options': year_options,
        'latest_year': str(latest_year),
        'status_options_region': status_options_region,
        'status_options_intracountry': status_options_intracountry,
        'status_options_single': status_options_single,
        'aggregation_level': aggregation_level,  # Store the current aggregation level
        'vessel_age': vessel_age  # Store the current vessel age
    }

    # Convert DataFrames to JSON for storage
    shipping_data = df_trades_shipping_region.to_json(date_format='iso', orient='split')
    shipping_balance = df_global_shipping_balance.to_json(date_format='iso', orient='split')
    shipping_balance_supply = df_global_shipping_balance_supply.to_json(date_format='iso', orient='split')
    intracountry_data = df_intracountry_trades.to_json(date_format='iso', orient='split')

    # Store timestamp
    refresh_timestamp = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    return shipping_data, shipping_balance, shipping_balance_supply, options_data, refresh_timestamp, intracountry_data


@callback(
    Output('last-refresh-indicator', 'children'),
    Input('refresh-timestamp-store', 'data'),
    prevent_initial_call=False
)
def update_refresh_time(timestamp):
    """Update the refresh time indicator."""
    if timestamp is None:
        return "No data loaded yet. Click 'Refresh data' to load data."
    return f"Last refreshed: {timestamp}"


# Update the update_visualizations callback to handle aggregation levels in chart formatting
@callback(
    Output('global-shipping-balance', 'figure'),
    Output('global-shipping-balance-supply', 'figure'),
    Output('region-dropdown', 'options'),
    Output('year-dropdown', 'options'),
    Output('year-dropdown', 'value'),
    Output('region-status-dropdown', 'options'),
    Output('region-status-dropdown', 'value'),
    Input('trades-shipping-data-store', 'data'),
    Input('shipping-balance-data-store', 'data'),
    Input('shipping-balance-supply-data-store', 'data'),
    Input('dropdown-options-store', 'data'),
    Input('region-dropdown', 'value'),
    Input('year-dropdown', 'value'),
    Input('region-status-dropdown', 'value'),
)
def update_visualizations(shipping_data, shipping_balance, shipping_balance_supply, dropdown_options, selected_regions,
                          selected_year, selected_statuses):
    """Update visualizations and tables based on selected filters."""
    # Check if data is available
    if shipping_data is None or shipping_balance is None or shipping_balance_supply is None or dropdown_options is None:
        raise PreventUpdate

    # Convert stored JSON back to DataFrames
    df_trades_shipping_region = pd.read_json(StringIO(shipping_data), orient='split')
    df_global_shipping_balance = pd.read_json(StringIO(shipping_balance), orient='split')
    df_global_shipping_balance_supply = pd.read_json(StringIO(shipping_balance_supply), orient='split')

    # Extract dropdown options
    region_options = dropdown_options['region_options']
    year_options = dropdown_options['year_options']
    status_options = dropdown_options['status_options_region']
    latest_year = dropdown_options['latest_year']
    # Get the aggregation level if available
    aggregation_level = dropdown_options.get('aggregation_level', 'monthly')

    # Set default values if needed
    if selected_year is None:
        selected_year = latest_year
    if selected_statuses is None:
        selected_statuses = ['All Statuses']



    # Create global shipping balance chart with professional formatting
    fig_global_shipping = make_subplots(specs=[[{"secondary_y": True}]])

    # Add main traces with professional colors
    fig_global_shipping.add_trace(
        go.Scatter(
            x=df_global_shipping_balance['date'],
            y=df_global_shipping_balance['total_active_ships'],
            name='Total Active Ships',
            mode='lines+markers',
            line=dict(color='#2E86C1', width=2),  # McKinsey blue
            marker=dict(size=6, color='#2E86C1'),
            opacity=0.9,
        ),
        secondary_y=False,
    )

    fig_global_shipping.add_trace(
        go.Scatter(
            x=df_global_shipping_balance['date'],
            y=df_global_shipping_balance['ships_demand_total'],
            name='Ships Demand Total',
            mode='lines+markers',
            line=dict(color='#22c55e', width=2),  # Professional green
            marker=dict(size=6, color='#22c55e'),
            opacity=0.9,
        ),
        secondary_y=False,
    )

    fig_global_shipping.add_trace(
        go.Bar(
            x=df_global_shipping_balance['date'],
            y=df_global_shipping_balance['net_new'],
            name='Net New',
            marker_color='rgba(239, 68, 68, 0.5)',  # Professional red with transparency
            marker_line_color='rgba(239, 68, 68, 0.8)',
            marker_line_width=1,
        ),
        secondary_y=True,
    )

    # Set the chart title to include the aggregation level
    aggregation_title = {
        'monthly': 'Monthly',
        'quarterly': 'Quarterly',
        'seasonal': 'Seasonal',
        'yearly': 'Yearly'
    }.get(aggregation_level, 'Monthly')

    # Professional chart layout following dash_style.md standards
    fig_global_shipping.update_layout(
        # No title - using external Demand View label
        title=None,
        
        # Professional Legend Positioning
        legend=dict(
            orientation='h',  # Horizontal layout
            yanchor='top',
            y=-0.08,  # Below chart
            xanchor='center',
            x=0.5,  # Centered
            bgcolor='rgba(255, 255, 255, 0)',  # Transparent
            bordercolor='rgba(255, 255, 255, 0)',
            borderwidth=0,
            font=dict(size=12, color='#4A4A4A'),
            itemsizing='constant',
            itemwidth=30
        ),
        
        # Professional Background and Margins
        plot_bgcolor='rgba(248, 249, 250, 0.5)',  # Subtle background
        paper_bgcolor='white',
        margin=dict(l=70, r=70, t=70, b=90),  # Reduced vertical spacing
        
        # Enhanced Interactivity
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='rgba(255, 255, 255, 0.95)',
            bordercolor='rgba(200, 200, 200, 0.8)',
            font=dict(size=11, color='#2C3E50'),
            align='left'
        ),
        
        # Bar gap for better visualization
        bargap=0.2,
        
        # Height to fit container
        height=400
    )

    # Professional Y-Axis Styling - Primary
    fig_global_shipping.update_yaxes(
        title=dict(text='Number of Ships', font=dict(size=13, color='#4A4A4A')),
        showgrid=True,
        gridcolor='rgba(200, 200, 200, 0.3)',
        gridwidth=0.5,
        linecolor='#CCCCCC',
        linewidth=1,
        tickfont=dict(size=11, color='#666666'),
        zeroline=True,
        zerolinecolor='rgba(150, 150, 150, 0.4)',
        zerolinewidth=1,
        secondary_y=False
    )
    
    # Professional Y-Axis Styling - Secondary
    fig_global_shipping.update_yaxes(
        title=dict(text='Net New Ships', font=dict(size=13, color='#4A4A4A')),
        showgrid=False,  # No grid for secondary axis
        linecolor='#CCCCCC',
        linewidth=1,
        tickfont=dict(size=11, color='#666666'),
        secondary_y=True
    )

    # Set appropriate x-axis formatting based on aggregation level
    if aggregation_level == 'monthly':
        tick_format = '%b %Y'
        dtick = "M3"  # Every 3 months
    elif aggregation_level == 'quarterly':
        tick_format = '%Y Q%q'
        dtick = "M3"  # Every 3 months
    elif aggregation_level == 'seasonal':
        tick_format = '%Y %b'  # Show month abbreviation (Jan/Jul)
        dtick = "M6"  # Every 6 months
    else:  # yearly
        tick_format = '%Y'
        dtick = "M12"  # Every 12 months

    # Professional X-Axis Styling
    fig_global_shipping.update_xaxes(
        title=dict(text='Date', font=dict(size=13, color='#4A4A4A')),
        tickformat=tick_format,
        tickangle=0,  # Angled for better readability with dates
        dtick=dtick,
        showgrid=True,
        gridcolor='rgba(200, 200, 200, 0.3)',
        gridwidth=0.5,
        linecolor='#CCCCCC',
        linewidth=1,
        tickfont=dict(size=11, color='#666666'),
        tickmode='auto',  # Changed from 'linear' to 'auto' to prevent tick overflow
        nticks=15  # Limit the number of ticks to prevent overcrowding
    )

    # Customize hover template based on aggregation level
    hover_templates = {
        'monthly': '%{x|%b %Y}<br><b>%{y:,.0f}</b><extra></extra>',
        'quarterly': '%{x|%Y Q%q}<br><b>%{y:,.0f}</b><extra></extra>',
        'seasonal': '%{x|%Y %b}<br><b>%{y:,.0f}</b><extra></extra>',
        'yearly': '%{x|%Y}<br><b>%{y:,.0f}</b><extra></extra>'
    }

    for trace in fig_global_shipping.data:
        trace.hovertemplate = hover_templates.get(aggregation_level, '%{x}<br><b>%{y:,.0f}</b><extra></extra>')

    # Create global shipping balance supply chart with professional formatting
    fig_global_shipping_supply = make_subplots(specs=[[{"secondary_y": True}]])

    # Add main traces for supply view with professional colors
    fig_global_shipping_supply.add_trace(
        go.Scatter(
            x=df_global_shipping_balance_supply['date'],
            y=df_global_shipping_balance_supply['total_active_ships'],
            name='Total Active Ships',
            mode='lines+markers',
            line=dict(color='#2E86C1', width=2),  # McKinsey blue
            marker=dict(size=6, color='#2E86C1'),
            opacity=0.9,
        ),
        secondary_y=False,
    )

    fig_global_shipping_supply.add_trace(
        go.Scatter(
            x=df_global_shipping_balance_supply['date'],
            y=df_global_shipping_balance_supply['ships_demand_total'],
            name='Ships Supply Total',
            mode='lines+markers',
            line=dict(color='#F7DC6F', width=2),  # Professional yellow/orange
            marker=dict(size=6, color='#F7DC6F'),
            opacity=0.9,
        ),
        secondary_y=False,
    )

    fig_global_shipping_supply.add_trace(
        go.Bar(
            x=df_global_shipping_balance_supply['date'],
            y=df_global_shipping_balance_supply['net_new'],
            name='Net New',
            marker_color='rgba(247, 220, 111, 0.5)',  # Professional yellow with transparency
            marker_line_color='rgba(247, 220, 111, 0.8)',
            marker_line_width=1,
        ),
        secondary_y=True,
    )

    # Professional chart layout following dash_style.md standards
    fig_global_shipping_supply.update_layout(
        # No title - using external Supply View label
        title=None,
        
        # Professional Legend Positioning
        legend=dict(
            orientation='h',  # Horizontal layout
            yanchor='top',
            y=-0.08,  # Below chart
            xanchor='center',
            x=0.5,  # Centered
            bgcolor='rgba(255, 255, 255, 0)',  # Transparent
            bordercolor='rgba(255, 255, 255, 0)',
            borderwidth=0,
            font=dict(size=12, color='#4A4A4A'),
            itemsizing='constant',
            itemwidth=30
        ),
        
        # Professional Background and Margins
        plot_bgcolor='rgba(248, 249, 250, 0.5)',  # Subtle background
        paper_bgcolor='white',
        margin=dict(l=70, r=70, t=70, b=90),  # Reduced vertical spacing
        
        # Enhanced Interactivity
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='rgba(255, 255, 255, 0.95)',
            bordercolor='rgba(200, 200, 200, 0.8)',
            font=dict(size=11, color='#2C3E50'),
            align='left'
        ),
        
        # Bar gap for better visualization
        bargap=0.2,
        
        # Height to fit container
        height=400
    )

    # Professional Y-Axis Styling - Primary
    fig_global_shipping_supply.update_yaxes(
        title=dict(text='Number of Ships', font=dict(size=13, color='#4A4A4A')),
        showgrid=True,
        gridcolor='rgba(200, 200, 200, 0.3)',
        gridwidth=0.5,
        linecolor='#CCCCCC',
        linewidth=1,
        tickfont=dict(size=11, color='#666666'),
        zeroline=True,
        zerolinecolor='rgba(150, 150, 150, 0.4)',
        zerolinewidth=1,
        secondary_y=False
    )
    
    # Professional Y-Axis Styling - Secondary
    fig_global_shipping_supply.update_yaxes(
        title=dict(text='Net New Ships', font=dict(size=13, color='#4A4A4A')),
        showgrid=False,  # No grid for secondary axis
        linecolor='#CCCCCC',
        linewidth=1,
        tickfont=dict(size=11, color='#666666'),
        secondary_y=True
    )

    # Professional X-Axis Styling
    fig_global_shipping_supply.update_xaxes(
        title=dict(text='Date', font=dict(size=13, color='#4A4A4A')),
        tickformat=tick_format,
        tickangle=0,  # Angled for better readability with dates
        dtick=dtick,
        showgrid=True,
        gridcolor='rgba(200, 200, 200, 0.3)',
        gridwidth=0.5,
        linecolor='#CCCCCC',
        linewidth=1,
        tickfont=dict(size=11, color='#666666'),
        tickmode='auto',  # Changed from 'linear' to 'auto' to prevent tick overflow
        nticks=15  # Limit the number of ticks to prevent overcrowding
    )

    for trace in fig_global_shipping_supply.data:
        trace.hovertemplate = hover_templates.get(aggregation_level, '%{x}<br><b>%{y:,.0f}</b><extra></extra>')


    return (
        fig_global_shipping,
        fig_global_shipping_supply,
        region_options,
        year_options,
        selected_year,
        status_options,
        selected_statuses,
    )


# Trade Analysis Visualization and Table Callback
@callback(
    Output('trade-analysis-visualization', 'figure'),
    Output('trade-analysis-table-container', 'children'),
    Output('vessel-types-info', 'children'),
    Input('trade-metric-selector', 'value'),
    Input('trades-shipping-data-store', 'data'),
    Input('region-dropdown', 'value'),
    Input('year-dropdown', 'value'),
    Input('region-status-dropdown', 'value'),
)
def update_trade_analysis_chart_and_table(selected_metric, shipping_data, selected_regions, selected_year, selected_statuses):
    """Update the trade analysis chart and table based on selected metric."""
    if not shipping_data:
        # Return empty figure and table
        empty_fig = go.Figure().update_layout(
            title="No data available",
            annotations=[dict(text="Please refresh data to load charts", 
                            x=0.5, y=0.5, showarrow=False)]
        )
        return empty_fig, "No data available", ""
    
    # Default to all statuses if none selected
    if selected_statuses is None:
        selected_statuses = ['All Statuses']
    
    try:
        # Convert JSON back to DataFrame using the same method as main callback
        from io import StringIO
        df_trades_shipping_region = pd.read_json(StringIO(shipping_data), orient='split')
        
        # Check if DataFrame is empty
        if df_trades_shipping_region.empty:
            return go.Figure().update_layout(
                title="No data to display",
                annotations=[dict(text="No data available", 
                                x=0.5, y=0.5, showarrow=False)]
            ), "No data available", ""
        
        # Don't apply additional filters - use the full dataset like the original callback
        # The filtering is handled by the table callbacks, not the chart callbacks
        
        # Determine metric based on selection
        metric_mapping = {
            'count_trades': ('count_trades', 'Count of Trades'),
            'ton_miles': ('sum_ton_miles', 'Ton Miles'),
            'median_delivery_days': ('median_delivery_days', 'Median Delivery Days'),
            'median_mileage_nautical_miles': ('median_mileage_nautical_miles', 'Median Mileage (Nautical Miles)'),
            'median_ton_miles': ('median_ton_miles', 'Median Ton Miles'),
            'median_utilization_rate': ('median_utilization_rate', 'Median Utilization Rate'),
            'median_cargo_destination_cubic_meters': ('median_cargo_destination_cubic_meters', 'Median Cargo Volume (m³)'),
            'median_vessel_capacity_cubic_meters': ('median_vessel_capacity_cubic_meters', 'Median Vessel Capacity (m³)')
        }
        
        metric, title_suffix = metric_mapping.get(selected_metric, ('count_trades', 'Count of Trades'))
        
        # Create the chart
        fig, vessel_footnote = create_stacked_bar_chart(
            df_trades_shipping_region,
            metric=metric,
            title_suffix=title_suffix,
            selected_statuses=selected_statuses,
            is_intracountry=False
        )
        
        # Create the table data using the same filtering logic as main callback
        filtered_df = df_trades_shipping_region.copy()
        
        # Apply filters
        if selected_regions and 'All Regions' not in selected_regions:
            filtered_df = filtered_df[filtered_df['origin_shipping_region'].isin(selected_regions)]
        
        if selected_year and selected_year != 'All Years':
            filtered_df = filtered_df[filtered_df['year'] == int(selected_year)]
        
        if selected_statuses and 'All Statuses' not in selected_statuses:
            filtered_df = filtered_df[filtered_df['status'].isin(selected_statuses)]
        
        # Create table data based on selected metric
        table_data = prepare_table_data(
            filtered_df,
            metric=metric,
            selected_statuses=selected_statuses,
            is_intracountry=False
        )
        
        # Create the table
        table = create_datatable(table_data, 'region_pair')
        
        return fig, table, vessel_footnote
        
    except Exception as e:
        print(f"Error updating trade analysis chart and table: {e}")
        error_fig = go.Figure().update_layout(
            title="Error loading chart",
            annotations=[dict(text=f"Error: {str(e)}", 
                            x=0.5, y=0.5, showarrow=False)]
        )
        return error_fig, f"Error loading table: {str(e)}", ""


@callback(
    Output('intracountry-count-visualization', 'figure'),
    Output('intracountry-tonmiles-visualization', 'figure'),
    Output('intracountry-year-dropdown', 'options'),
    Output('intracountry-year-dropdown', 'value'),
    Output('intracountry-status-dropdown', 'options'),
    Output('intracountry-status-dropdown', 'value'),
    Output('intracountry-count-table-container', 'children'),
    Output('intracountry-tonmiles-table-container', 'children'),
    Input('intracountry-data-store', 'data'),
    Input('dropdown-options-store', 'data'),
    Input('intracountry-year-dropdown', 'value'),
    Input('intracountry-status-dropdown', 'value'),
    prevent_initial_call=False
)
def update_intracountry_visualizations(intracountry_data, dropdown_options, selected_year, selected_statuses):
    """Update intracountry visualizations and tables based on selected filters."""
    # Check if data is available
    if intracountry_data is None or dropdown_options is None:
        raise PreventUpdate

    # Convert stored JSON back to DataFrame
    df_intracountry_trades = pd.read_json(StringIO(intracountry_data), orient='split')

    # Extract dropdown options
    year_options = dropdown_options['year_options']
    status_options = dropdown_options['status_options_intracountry']
    latest_year = dropdown_options['latest_year']

    # Set default values if needed
    if selected_year is None:
        selected_year = latest_year
    if selected_statuses is None:
        selected_statuses = ['All Statuses']

    # Create visualizations
    fig_intracountry_count, _ = create_stacked_bar_chart(
        df_intracountry_trades,
        metric='count_trades',
        title_suffix='Count of Trades',
        selected_statuses=selected_statuses,
        is_intracountry=True
    )

    fig_intracountry_tonmiles, _ = create_stacked_bar_chart(
        df_intracountry_trades,
        metric='sum_ton_miles',
        title_suffix='Ton Miles',
        selected_statuses=selected_statuses,
        is_intracountry=True
    )

    # Prepare data for tables
    count_table_data = prepare_table_data(
        df_intracountry_trades,
        'count_trades',
        selected_year=selected_year,
        selected_statuses=selected_statuses,
        is_intracountry=True
    )

    ton_miles_table_data = prepare_table_data(
        df_intracountry_trades,
        'sum_ton_miles',
        selected_year=selected_year,
        selected_statuses=selected_statuses,
        is_intracountry=True
    )

    # Create data tables
    intracountry_count_table = create_datatable(count_table_data, 'origin_country_name')
    intracountry_tonmiles_table = create_datatable(ton_miles_table_data, 'origin_country_name')

    return (
        fig_intracountry_count,
        fig_intracountry_tonmiles,
        year_options,
        selected_year,
        status_options,
        selected_statuses,
        intracountry_count_table,
        intracountry_tonmiles_table
    )


def prepare_custom_metrics_data(df, metric, selected_year=None, selected_status=None, region_direction=None,
                                is_intracountry=False):
    """
    Prepare data for custom metrics table with vessel types as columns.

    Args:
        df: DataFrame with trade data
        metric: The column to display (e.g., 'median_delivery_days')
        selected_year: Year to filter by (defaults to latest)
        selected_status: Status value to filter by (defaults to 'laden')
        region_direction: How to handle regions for region metrics
        is_intracountry: Flag indicating if using intracountry data (which has origin_country_name)

    Returns:
        DataFrame formatted for display in a DataTable
    """
    # Filter data for years 2019 and later
    filtered_df = df[df['year'] >= 2019].copy()

    # Get the latest year if not specified
    if not selected_year or selected_year == "All Years":
        selected_year = filtered_df['year'].max()
    else:
        selected_year = int(selected_year)

    # Filter by year
    filtered_df = filtered_df[filtered_df['year'] == selected_year]

    # Filter by status if specified
    if selected_status and selected_status != "All Statuses":
        filtered_df = filtered_df[filtered_df['status'] == selected_status]

    # Check the type of analysis we're doing
    if region_direction:
        # For region metrics using trades_shipping_region data
        if region_direction == 'origin_to_destination':
            # Create a combined field for origin-destination pairs
            filtered_df['region_pair'] = filtered_df['origin_shipping_region'] + ' → ' + filtered_df[
                'destination_shipping_region']
            index_field = 'region_pair'
        elif region_direction == 'origin':
            index_field = 'origin_shipping_region'
        elif region_direction == 'destination':
            index_field = 'destination_shipping_region'
        else:
            # Default to region_pair if invalid value
            filtered_df['region_pair'] = filtered_df['origin_shipping_region'] + ' → ' + filtered_df[
                'destination_shipping_region']
            index_field = 'region_pair'
        # Select only the required columns
        keep_cols = ['year', index_field, 'vessel_type', 'status', metric]
    elif is_intracountry:
        # For intracountry data with origin_country_name
        index_field = 'origin_country_name'
        # Select only required columns
        keep_cols = ['year', index_field, 'vessel_type', 'status', metric]
    else:
        # For vessel metrics: We're analyzing by vessel type, so no region needed
        index_field = 'vessel_type'
        # Select only required columns
        keep_cols = ['year', 'vessel_type', 'status', metric]
    # Keep only the required columns that exist in the DataFrame
    existing_cols = [col for col in keep_cols if col in filtered_df.columns]
    filtered_df = filtered_df[existing_cols]

    # For vessel metrics, handle differently
    if not region_direction and not is_intracountry:
        # For vessel metrics - aggregate the data by vessel type
        agg_data = filtered_df.groupby(['vessel_type']).agg({
            metric: 'mean'
        }).reset_index()
        # Sort values by vessel_type
        pivot_table = agg_data.sort_values('vessel_type')
    else:
        # For region or country metrics - pivot with vessel types as columns
        try:
            pivot_table = filtered_df.pivot_table(
                index=[index_field, 'year', 'status'],
                columns='vessel_type',
                values=metric,
                aggfunc='mean'  # Using mean for aggregation if there are multiple entries
            ).reset_index()
            # Sort the data by the index field
            pivot_table = pivot_table.sort_values(index_field)
        except KeyError as e:
            # Handle case where expected columns aren't found
            print(f"KeyError in pivot_table: {e}")
            # Create a simple DataFrame to show the error
            pivot_table = pd.DataFrame({
                'Error': [f"Missing column: {e}"]
            })

    return pivot_table


@callback(
    Output('custom-metrics-table-title', 'children'),
    Output('custom-metrics-table-container', 'children'),
    Input('intracountry-data-store', 'data'),
    Input('dropdown-options-store', 'data'),
    Input('metrics-year-dropdown', 'value'),
    Input('metrics-status-dropdown', 'value'),
    Input('metric-dropdown', 'value'),
    prevent_initial_call=False
)
def update_custom_metrics_table(intracountry_data, dropdown_options, selected_year, selected_status, selected_metric):
    """Update custom metrics table based on selected filters."""
    # Check if data is available
    if intracountry_data is None or dropdown_options is None:
        raise PreventUpdate

    # Convert stored JSON back to DataFrame
    df_intracountry_trades = pd.read_json(StringIO(intracountry_data), orient='split')

    # Extract dropdown options
    year_options = dropdown_options['year_options']
    status_options = dropdown_options['status_options_intracountry']
    latest_year = dropdown_options['latest_year']

    # Set default values if needed
    if selected_year is None:
        selected_year = latest_year

    # Set default status to 'laden' if not specified
    if selected_status is None:
        # Find 'laden' in status options
        laden_option = next((option['value'] for option in status_options
                             if option['value'].lower() == 'laden'), None)
        if laden_option:
            selected_status = laden_option
        else:
            # If 'laden' is not available, use the first status
            selected_status = status_options[0]['value'] if status_options else None

    # Get human-readable metric name
    metric_name = next((option['label'] for option in [
        {'label': 'Median Delivery Days', 'value': 'median_delivery_days'},
        # {'label': 'Mean Delivery Days', 'value': 'mean_delivery_days'},
        # {'label': 'Std Delivery Days', 'value': 'std_delivery_days'},
        # {'label': 'Mean Mileage (Nautical Miles)', 'value': 'mean_mileage_nautical_miles'},
        {'label': 'Median Mileage (Nautical Miles)', 'value': 'median_mileage_nautical_miles'},
        # {'label': 'Std Mileage (Nautical Miles)', 'value': 'std_mileage_nautical_miles'},
        {'label': 'Median Ton Miles', 'value': 'median_ton_miles'},
        # {'label': 'Std Ton Miles', 'value': 'std_ton_miles'},
        # {'label': 'Mean Utilization Rate', 'value': 'mean_utilization_rate'},
        {'label': 'Median Utilization Rate', 'value': 'median_utilization_rate'},
        # {'label': 'Mean Cargo Volume (m³)', 'value': 'mean_cargo_destination_cubic_meters'},
        {'label': 'Median Cargo Volume (m³)', 'value': 'median_cargo_destination_cubic_meters'},
        # {'label': 'Std Cargo Volume (m³)', 'value': 'std_cargo_destination_cubic_meters'},
        # {'label': 'Mean Vessel Capacity (m³)', 'value': 'mean_vessel_capacity_cubic_meters'},
        {'label': 'Median Vessel Capacity (m³)', 'value': 'median_vessel_capacity_cubic_meters'},
        {'label': 'Count of Trades', 'value': 'count_trades'}
    ] if option['value'] == selected_metric), selected_metric)

    # Prepare table title
    status_display = selected_status if selected_status != "All Statuses" else "All Statuses"
    table_title = f"{metric_name} by Vessel Type ({selected_year}, {status_display})"

    try:
        # Prepare data for table - note we specify is_intracountry=True
        custom_metrics_data = prepare_custom_metrics_data(
            df_intracountry_trades,
            selected_metric,
            selected_year,
            selected_status,
            region_direction=None,
            is_intracountry=True  # Specify that we're using intracountry data
        )
        # Check if we got data
        if custom_metrics_data.empty:
            return table_title, html.Div("No data available for the selected filters.")
        # Handle error case
        if 'Error' in custom_metrics_data.columns:
            return table_title, html.Div(f"Error: {custom_metrics_data['Error'].values[0]}")
        # Create columns for the table
        columns = []
        for col in custom_metrics_data.columns:
            if col in ['vessel_type', 'origin_country_name', 'year', 'status']:
                display_name = {
                    'vessel_type': 'Vessel Type',
                    'origin_country_name': 'Origin Country',
                    'year': 'Year',
                    'status': 'Status'
                }.get(col, col.capitalize())
                columns.append({"name": display_name, "id": col})
            else:
                # Other columns (the metric or vessel types)
                columns.append({
                    "name": col,
                    "id": col,
                    "type": "numeric",
                    "format": Format(
                        precision=2,
                        scheme=Scheme.fixed,
                        group=Group.yes,
                        group_delimiter=',',
                        decimal_delimiter='.'
                    )
                })

        # Create the table
        custom_metrics_table = dash_table.DataTable(
            id='custom-metrics-table',
            columns=columns,
            data=custom_metrics_data.to_dict('records'),
            style_table={'overflowX': 'auto'},
            style_cell={
                'textAlign': 'left',
                'padding': '5px',
                'minWidth': '100px',
                'maxWidth': '300px',
                'whiteSpace': 'normal',
            },
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            },
            page_size=25,
            sort_action='native',
            filter_action='native',
            fill_width=False,
            export_format='xlsx',
            export_headers='display',
            export_columns='visible'
        )
        return table_title, custom_metrics_table
    except Exception as e:
        print(f"Error in update_custom_metrics_table: {str(e)}")
        return table_title, html.Div(f"Error: {str(e)}")


# First callback - just handles the dropdowns
@callback(
    Output('region-metrics-year-dropdown', 'options'),
    Output('region-metrics-year-dropdown', 'value'),
    Output('region-metrics-status-dropdown', 'options'),
    Output('region-metrics-status-dropdown', 'value'),
    Input('dropdown-options-store', 'data'),
    prevent_initial_call=False
)
def update_region_metrics_dropdowns(dropdown_options):
    """Update region metrics dropdown options."""
    # Check if data is available
    if dropdown_options is None:
        raise PreventUpdate

    # Extract dropdown options
    year_options = dropdown_options['year_options']
    status_options = dropdown_options['status_options_region']
    latest_year = dropdown_options['latest_year']

    # Set default values
    selected_year = latest_year
    # Set default status to 'laden' if available
    laden_option = next((option['value'] for option in status_options
                         if option['value'].lower() == 'laden'), None)
    if laden_option:
        selected_status = laden_option
    else:
        # If 'laden' is not available, use the first status
        selected_status = status_options[0]['value'] if status_options else None

    return year_options, selected_year, status_options, selected_status


# Second callback - handles the table and title
@callback(
    Output('region-custom-metrics-table-title', 'children'),
    Output('region-custom-metrics-table-container', 'children'),
    Input('trades-shipping-data-store', 'data'),
    Input('region-metrics-year-dropdown', 'value'),
    Input('region-metrics-status-dropdown', 'value'),
    Input('region-metric-dropdown', 'value'),
    # Input('region-direction', 'value'),
    prevent_initial_call=False
)
def update_region_metrics_table(shipping_data, selected_year, selected_status,
                                selected_metric#, region_direction
                                ):
    """Update region custom metrics table based on selected filters."""
    # Check if data is available
    if shipping_data is None:
        raise PreventUpdate

    try:
        # Convert stored JSON back to DataFrame
        df_trades_shipping_region = pd.read_json(StringIO(shipping_data), orient='split')
        # Get human-readable metric name
        metric_options = [
            {'label': 'Median Delivery Days', 'value': 'median_delivery_days'},
            # {'label': 'Mean Delivery Days', 'value': 'mean_delivery_days'},
            # {'label': 'Mean Mileage (Nautical Miles)', 'value': 'mean_mileage_nautical_miles'},
            {'label': 'Median Mileage (Nautical Miles)', 'value': 'median_mileage_nautical_miles'},
            {'label': 'Median Ton Miles', 'value': 'median_ton_miles'},
            # {'label': 'Mean Utilization Rate', 'value': 'mean_utilization_rate'},
            {'label': 'Median Utilization Rate', 'value': 'median_utilization_rate'},
            # {'label': 'Mean Cargo Volume (m³)', 'value': 'mean_cargo_destination_cubic_meters'},
            {'label': 'Median Cargo Volume (m³)', 'value': 'median_cargo_destination_cubic_meters'},
            # {'label': 'Mean Vessel Capacity (m³)', 'value': 'mean_vessel_capacity_cubic_meters'},
            {'label': 'Median Vessel Capacity (m³)', 'value': 'median_vessel_capacity_cubic_meters'},
            {'label': 'Count of Trades', 'value': 'count_trades'},
        ]

        metric_name = next((option['label'] for option in metric_options
                            if option['value'] == selected_metric), selected_metric)

        # Get region type description
        region_type_desc = 'Shipping Regions'
        #     {
        #     'origin_to_destination': 'Shipping Region Pairs',
        #     'origin': 'Origin Shipping Regions',
        #     'destination': 'Destination Shipping Regions'
        # }.get(region_direction, 'Shipping Regions')
        region_direction= 'origin_to_destination'

        # Prepare table title
        status_display = selected_status if selected_status != "All Statuses" else "All Statuses"
        table_title = f"{metric_name} by {region_type_desc} and Vessel Type ({selected_year}, {status_display})"

        # Filter data for years 2019 and later
        filtered_df = df_trades_shipping_region[df_trades_shipping_region['year'] >= 2019].copy()

        # Get the latest year if not specified
        if not selected_year or selected_year == "All Years":
            selected_year = filtered_df['year'].max()
        else:
            selected_year = int(selected_year)

        # Filter by year
        filtered_df = filtered_df[filtered_df['year'] == selected_year]

        # Filter by status if specified
        if selected_status and selected_status != "All Statuses":
            filtered_df = filtered_df[filtered_df['status'] == selected_status]

        # Handle region direction
        if region_direction == 'origin_to_destination':
            # Create a combined field for origin-destination pairs
            filtered_df['region_pair'] = filtered_df['origin_shipping_region'] + ' → ' + filtered_df[
                'destination_shipping_region']
            index_field = 'region_pair'
        elif region_direction == 'origin':
            index_field = 'origin_shipping_region'
        elif region_direction == 'destination':
            index_field = 'destination_shipping_region'
        else:
            # Default to region_pair if invalid value
            filtered_df['region_pair'] = filtered_df['origin_shipping_region'] + ' → ' + filtered_df[
                'destination_shipping_region']
            index_field = 'region_pair'

        # Directly aggregate the data without using the helper function
        agg_data = filtered_df.groupby([index_field, 'year', 'season', 'quarter', 'status', 'vessel_type'])[
            selected_metric].mean().reset_index()
        # Pivot the data for the table
        pivot_table = agg_data.pivot_table(
            index=[index_field, 'year', 'season', 'quarter', 'status'],
            columns='vessel_type',
            values=selected_metric,
            aggfunc='mean'
        ).reset_index()
        # Create columns for the table
        columns = []
        # Add index columns first
        for col in [index_field, 'year', 'season', 'quarter', 'status']:
            display_name = {
                'region_pair': 'Region Pair',
                'origin_shipping_region': 'Origin Region',
                'destination_shipping_region': 'Destination Region',
                'year': 'Year',
                'season': 'Season',
                'quarter': 'Quarter',
                'status': 'Status'
            }.get(col, col.capitalize())
            columns.append({"name": display_name, "id": col})
        # Add vessel type columns
        for col in pivot_table.columns:
            if col not in [index_field, 'year', 'season', 'quarter', 'status']:
                columns.append({
                    "name": col,
                    "id": col,
                    "type": "numeric",
                    "format": Format(
                        precision=2,
                        scheme=Scheme.fixed,
                        group=Group.yes,
                        group_delimiter=',',
                        decimal_delimiter='.'
                    )
                })

        # Create the table
        custom_metrics_table = dash_table.DataTable(
            id='region-custom-metrics-table',
            columns=columns,
            data=pivot_table.to_dict('records'),
            style_table={'overflowX': 'auto'},
            style_cell={
                'textAlign': 'left',
                'padding': '5px',
                'minWidth': '100px',
                'maxWidth': '300px',
                'whiteSpace': 'normal',
            },
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            },
            page_size=25,
            sort_action='native',
            filter_action='native',
            fill_width=False,
            export_format='xlsx',
            export_headers='display',
            export_columns='visible'
        )

        return table_title, custom_metrics_table
    except Exception as e:
        # Print for debugging
        import traceback
        traceback_str = traceback.format_exc()
        print(f"Error in update_region_metrics_table: {str(e)}")
        print(traceback_str)
        # Return error message to the user
        return "Error in Custom Metrics Analysis", html.Div([
            html.P(f"An error occurred: {str(e)}"),
            html.Pre(traceback_str)
        ])
 