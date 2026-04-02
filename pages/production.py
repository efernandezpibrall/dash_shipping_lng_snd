from dash import html, dcc, dash_table, callback, Input, Output, State
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
from io import BytesIO
import os
import sys
from dash.dash_table.Format import Format, Scheme
from dash.exceptions import PreventUpdate

# Add project root to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.insert(0, project_root)

from fundamentals.terminals.scenario_utils import get_available_scenarios
from utils.export_flow_data import (
    build_export_flow_matrix,
    default_selected_countries,
    get_available_countries,
)
from utils.table_styles import StandardTableStyleManager, TABLE_COLORS
from pages.terminals import (
    PRIMARY_COLORS,
    convert_to_mcmd,
    engine,
    fetch_volume_data,
    hex_to_rgb,
)


def _create_empty_volume_figure(message):
    """Create an empty state figure for the production page."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=18, family='Arial', color='#64748b')
    )
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=500,
        margin=dict(l=80, r=80, t=80, b=80),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )
    return fig


def _create_empty_table_state(message):
    """Return a consistent empty state for the country table."""
    return html.Div(message, className="balance-empty-state")


def _get_total_output_label(selected_unit='mtpa'):
    """Return the total-column label matching the active unit."""
    return 'Total MTPA' if selected_unit == 'mtpa' else 'Total Mcm/d'


def _format_table_cell_value(value):
    if pd.isna(value):
        return ""

    if isinstance(value, (int, float)):
        return f"{float(value):.2f}"

    return str(value)


def _build_volume_table_column_styles(df):
    """Create responsive widths so wide country matrices remain readable."""
    column_styles = []
    column_weights = {}
    column_min_widths = {}

    for column_name in df.columns:
        header_length = len(str(column_name))
        value_lengths = df[column_name].map(_format_table_cell_value).map(len)
        max_length = max([header_length] + value_lengths.tolist()) if not df.empty else header_length

        if column_name == 'Month':
            column_weights[column_name] = max(8, min(max_length, 12))
            column_min_widths[column_name] = 92
        elif str(column_name).startswith('Total '):
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

        if column_name == 'Month':
            style_entry["textAlign"] = "left"

        column_styles.append(style_entry)

    return column_styles


def _prepare_volume_country_dataframe(
    scenario='base_view',
    selected_unit='mtpa',
    new_capacity_only=False,
    start_year=2025,
    end_year=2040,
    breakdown='country',
):
    """Fetch the production-page data in a month/group_name shape."""
    raw_df = fetch_volume_data(
        start_year=start_year,
        end_year=end_year,
        breakdown=breakdown,
        new_capacity_only=new_capacity_only,
        selected_countries=None,
        scenario=scenario,
    )

    if raw_df.empty:
        return pd.DataFrame(columns=['month', 'country_name', 'total_output'])

    country_df = raw_df[['year', 'month', 'group_name', 'total_output']].copy()
    country_df = country_df.rename(columns={'group_name': 'country_name'})
    country_df['month'] = pd.to_datetime(country_df[['year', 'month']].assign(day=1))

    if selected_unit == 'mcmd':
        country_df['total_output'] = convert_to_mcmd(country_df['total_output'])

    return country_df[['month', 'country_name', 'total_output']]


def _build_direct_pivot_matrix(raw_df, selected_unit='mtpa'):
    """Build a month-by-group pivot for plant/train breakdowns (no Rest of the World logic)."""
    total_column_label = _get_total_output_label(selected_unit)
    if raw_df.empty:
        return pd.DataFrame(columns=['Month', total_column_label])

    pivot = raw_df.pivot_table(
        index='month', columns='country_name', values='total_output', aggfunc='sum'
    ).fillna(0)

    col_totals = pivot.sum().sort_values(ascending=False)
    pivot = pivot[col_totals.index]

    pivot[total_column_label] = pivot.sum(axis=1)
    pivot = pivot[pivot[total_column_label] > 0]
    pivot.index = pivot.index.strftime('%Y-%m')

    result = pivot.reset_index().rename(columns={'month': 'Month'})
    cols = ['Month'] + [c for c in result.columns if c not in {'Month', total_column_label}] + [total_column_label]
    return result[cols]


def _get_available_volume_countries(raw_df):
    """Reuse the shared balance-page country ordering for default selections."""
    if raw_df.empty:
        return []

    available_source_df = raw_df.rename(columns={'total_output': 'total_mmtpa'})
    return get_available_countries(
        [available_source_df[['month', 'country_name', 'total_mmtpa']]]
    )


def _build_volume_country_matrix(
    raw_df,
    selected_countries,
    other_countries_mode='rest_of_world',
    selected_unit='mtpa',
):
    """Build the month-by-country matrix shown below the chart."""
    total_column_label = _get_total_output_label(selected_unit)

    if raw_df.empty:
        return pd.DataFrame(columns=['Month', total_column_label])

    matrix_source_df = raw_df.rename(columns={'total_output': 'total_mmtpa'})
    matrix_df = build_export_flow_matrix(
        matrix_source_df[['month', 'country_name', 'total_mmtpa']],
        selected_countries,
        other_countries_mode,
    )

    if matrix_df.empty:
        return pd.DataFrame(columns=['Month', total_column_label])

    matrix_df = matrix_df.rename(columns={'Total MMTPA': total_column_label})
    country_columns = [
        column_name
        for column_name in matrix_df.columns
        if column_name not in {'Month', total_column_label}
    ]

    ordered_columns = ['Month'] + country_columns + [total_column_label]
    return matrix_df[ordered_columns]


def _create_volume_country_table(table_id, df):
    """Create a formatted table for the month-by-country matrix."""
    if df.empty:
        return _create_empty_table_state("No data available for the current selection.")

    base_config = StandardTableStyleManager.get_base_datatable_config()
    numeric_columns = [column for column in df.columns if column != 'Month']

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

    style_data_conditional = list(base_config['style_data_conditional'])
    style_data_conditional.append(
        {
            "if": {"column_id": "Month"},
            "backgroundColor": "#f8fafc",
            "fontWeight": "600",
            "color": TABLE_COLORS['text_primary'],
        }
    )

    for column_name in numeric_columns:
        if str(column_name).startswith('Total '):
            style_data_conditional.append(
                {
                    "if": {"column_id": column_name},
                    "backgroundColor": "#edf6fd",
                    "fontWeight": "700",
                    "color": TABLE_COLORS['primary_dark'],
                }
            )
        elif column_name == 'Rest of the World':
            style_data_conditional.append(
                {
                    "if": {"column_id": column_name},
                    "backgroundColor": "#f8f9fa",
                    "color": TABLE_COLORS['text_secondary'],
                }
            )

    return dash_table.DataTable(
        id=table_id,
        columns=columns,
        data=df.to_dict('records'),
        sort_action='native',
        page_action='none',
        fill_width=True,
        fixed_rows={"headers": True},
        fixed_columns={"headers": True, "data": 1},
        style_table={
            "overflowX": "auto",
            "overflowY": "auto",
            "maxHeight": "560px",
            "width": "100%",
            "minWidth": "100%",
            "marginTop": "20px",
        },
        style_header=base_config['style_header'],
        style_cell={
            **base_config['style_cell'],
            "minWidth": "72px",
            "width": "72px",
            "maxWidth": "none",
            "border": f"1px solid {TABLE_COLORS['border_light']}",
            "padding": "6px 8px",
        },
        style_cell_conditional=_build_volume_table_column_styles(df),
        style_data_conditional=style_data_conditional,
    )


def create_volume_country_area_chart(matrix_df, selected_unit='mtpa'):
    """Create the stacked area chart using the visible country columns."""
    total_column_label = _get_total_output_label(selected_unit)
    unit_label = 'MTPA' if selected_unit == 'mtpa' else 'Mcm/d'
    if matrix_df.empty:
        return _create_empty_volume_figure("No volume data available")

    country_columns = [
        column_name
        for column_name in matrix_df.columns
        if column_name not in {'Month', total_column_label}
    ]

    if not country_columns:
        return _create_empty_volume_figure(
            "Select at least one country or switch to Rest of the World mode."
        )

    plot_df = matrix_df.copy()
    plot_df['date'] = pd.to_datetime(plot_df['Month'] + '-01')
    pivot_df = plot_df.set_index('date')[country_columns]
    pivot_df = pivot_df[pivot_df.sum(axis=1) > 0]

    if pivot_df.empty:
        return _create_empty_volume_figure("No volume data available")

    column_totals = pivot_df.sum().sort_values(ascending=False)
    pivot_df = pivot_df[column_totals.index]

    fig = go.Figure()
    for group_name in pivot_df.columns:
        color = '#94A3B8' if group_name == 'Rest of the World' else PRIMARY_COLORS.get(group_name, '#666666')
        rgb = hex_to_rgb(color)

        fig.add_trace(go.Scatter(
            x=pivot_df.index,
            y=pivot_df[group_name],
            mode='lines',
            name=group_name,
            line=dict(width=0.5, color=color),
            fill='tonexty',
            fillcolor=f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.7)',
            hovertemplate=f'<b>{group_name}</b><br>Date: %{{x|%b %Y}}<br>Output: %{{y:.1f}} {unit_label}<extra></extra>',
            stackgroup='one'
        ))

    start_date = pivot_df.index.min()
    end_date = pivot_df.index.max()

    fig.update_layout(
        title={
            'text': f'Cumulative Monthly LNG Output by Country ({unit_label}) | {start_date.year}-{end_date.year}',
            'font': {'size': 18, 'family': 'Arial', 'color': '#333333'},
            'x': 0.5,
            'xanchor': 'center',
            'y': 0.95,
            'yanchor': 'top'
        },
        xaxis=dict(
            title='',
            range=[start_date, end_date],
            type='date',
            tickformat='%b\n%Y',
            dtick='M3',
            tickfont=dict(size=9, family='Arial', color='#333333'),
            showgrid=True,
            gridcolor='#E8E8E8',
            showline=True,
            linewidth=1,
            linecolor='#CCCCCC'
        ),
        yaxis=dict(
            title=dict(
                text=f'Monthly Output ({unit_label})',
                font=dict(size=12, family='Arial', color='#333333')
            ),
            showgrid=True,
            gridcolor='#E8E8E8',
            showline=True,
            linewidth=1,
            linecolor='#CCCCCC',
            zeroline=False
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=500,
        margin=dict(l=80, r=180, t=80, b=80),
        hovermode='x unified',
        legend=dict(
            orientation='v',
            yanchor='middle',
            y=0.5,
            xanchor='left',
            x=1.02,
            font=dict(size=10, family='Arial'),
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='#CCCCCC',
            borderwidth=1
        )
    )

    return fig


layout = html.Div([
    html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Scenario", className="filter-group-header"),
                            html.Label("View:", className="filter-label"),
                            dcc.Dropdown(
                                id='capacity-scenario-dropdown',
                                options=[{'label': s, 'value': s} for s in get_available_scenarios(engine)],
                                value='base_view',
                                clearable=False,
                                className='filter-dropdown',
                                style={'width': '100%'}
                            ),
                            html.A(
                                html.Button(
                                    'Manage Adjustments',
                                    style={
                                        'padding': '5px 12px',
                                        'backgroundColor': '#2E86C1',
                                        'color': 'white',
                                        'border': 'none',
                                        'borderRadius': '4px',
                                        'cursor': 'pointer',
                                        'fontWeight': 'bold',
                                        'fontSize': '12px',
                                        'marginTop': '4px'
                                    }
                                ),
                                href='/terminal_adjustments',
                            ),
                        ],
                        className="filter-section filter-section-destination",
                    ),
                    html.Div(
                        [
                            html.Div("Unit of Measure", className="filter-group-header"),
                            html.Label("Display:", className="filter-label"),
                            dcc.Dropdown(
                                id='capacity-unit-dropdown',
                                options=[
                                    {'label': 'MTPA (Million Tonnes Per Annum)', 'value': 'mtpa'},
                                    {'label': 'Mcm/d (Million Cubic Meters per Day)', 'value': 'mcmd'}
                                ],
                                value='mtpa',
                                clearable=False,
                                className='filter-dropdown',
                                style={'width': '100%'}
                            ),
                        ],
                        className="filter-section filter-section-origin",
                    ),
                    html.Div(
                        [
                            html.Div("Scope", className="filter-group-header"),
                            html.Label("View:", className="filter-label"),
                            dcc.Checklist(
                                id='capacity-new-capacity-checkbox',
                                options=[{'label': ' New capacity only', 'value': 'new_only'}],
                                value=[],
                                style={'fontSize': '14px', 'fontWeight': '600'}
                            ),
                        ],
                        className="filter-section filter-section-destination",
                    ),
                    html.Div(
                        [
                            html.Div("Group By", className="filter-group-header"),
                            html.Label("Level:", className="filter-label"),
                            dcc.Dropdown(
                                id='capacity-breakdown-dropdown',
                                options=[
                                    {'label': 'Country', 'value': 'country'},
                                    {'label': 'Plant', 'value': 'project'},
                                    {'label': 'Train', 'value': 'train'},
                                ],
                                value='country',
                                clearable=False,
                                className='filter-dropdown',
                                style={'width': '100%'}
                            ),
                        ],
                        className="filter-section filter-section-volume",
                    ),
                    html.Div(
                        [
                            html.Div("Country Columns", className="filter-group-header"),
                            html.Label("Countries:", className="filter-label"),
                            dcc.Dropdown(
                                id='capacity-country-columns-dropdown',
                                options=[],
                                value=None,
                                multi=True,
                                placeholder='Select countries to keep as separate columns',
                                className='filter-dropdown',
                                style={'width': '100%'}
                            ),
                        ],
                        id='capacity-country-columns-section',
                        className="filter-section filter-section-origin-exp",
                    ),
                    html.Div(
                        [
                            html.Div("Other Countries", className="filter-group-header"),
                            html.Label("Handling:", className="filter-label"),
                            dcc.RadioItems(
                                id='capacity-other-country-mode',
                                options=[
                                    {
                                        'label': 'Include as Rest of the World',
                                        'value': 'rest_of_world',
                                    },
                                    {
                                        'label': 'Exclude from the chart and table',
                                        'value': 'exclude',
                                    },
                                ],
                                value='rest_of_world',
                                className='balance-radio-group',
                                labelStyle={'display': 'inline-flex', 'alignItems': 'center'},
                                inputStyle={'marginRight': '6px'},
                            ),
                        ],
                        id='capacity-other-country-section',
                        className="filter-section filter-section-volume",
                    ),
                    html.Div(
                        [
                            html.Div("Export", className="filter-group-header"),
                            html.Label("Visible matrix:", className="filter-label"),
                            html.Button(
                                'Export to Excel',
                                id='capacity-export-excel-button',
                                n_clicks=0,
                                style={
                                    'padding': '8px 16px',
                                    'backgroundColor': '#2E86C1',
                                    'color': 'white',
                                    'border': 'none',
                                    'borderRadius': '4px',
                                    'cursor': 'pointer',
                                    'fontWeight': 'bold',
                                    'fontSize': '14px'
                                }
                            ),
                            dcc.Download(id='capacity-download-excel')
                        ],
                        className="filter-section filter-section-analysis",
                    ),
                ],
                className="filter-bar-grouped",
            )
        ],
        className="professional-section-header",
        style={'margin': '0'}
    ),
    html.Div([
        html.Div([
            html.Label("Year Range:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
            dcc.RangeSlider(
                id='capacity-year-range-slider',
                min=2000,
                max=2055,
                step=1,
                value=[2025, 2040],
                marks={year: str(year) for year in range(2000, 2056, 5)},
                tooltip={"placement": "bottom", "always_visible": True},
                className='year-range-slider'
            )
        ], style={'width': '80%', 'margin': '0 auto', 'marginBottom': '20px'})
    ], style={'textAlign': 'center'}),
    html.Div([
        dcc.Graph(
            id='capacity-volume-area-chart',
            figure=_create_empty_volume_figure("Loading capacity data..."),
            config={'displayModeBar': True, 'displaylogo': False},
            style={'height': '100%'}
        )
    ], style={'marginTop': '20px'}),
    html.Div(
        id='capacity-country-table-container',
        children=_create_empty_table_state("Loading capacity data..."),
        style={'marginTop': '20px'}
    )
])


@callback(
    Output('capacity-country-columns-dropdown', 'options'),
    Output('capacity-country-columns-dropdown', 'value'),
    Output('capacity-country-columns-section', 'style'),
    Output('capacity-other-country-section', 'style'),
    Input('capacity-scenario-dropdown', 'value'),
    Input('capacity-new-capacity-checkbox', 'value'),
    Input('capacity-year-range-slider', 'value'),
    Input('capacity-breakdown-dropdown', 'value'),
    Input('global-refresh-button', 'n_clicks'),
    State('capacity-country-columns-dropdown', 'value')
)
def populate_capacity_country_columns(
    scenario,
    new_capacity_checkbox,
    year_range,
    breakdown,
    n_clicks,
    current_selection,
):
    """Populate the country-columns selector; hide country-only sections for plant/train."""
    if not scenario:
        raise PreventUpdate

    is_country = (breakdown or 'country') == 'country'
    hidden = {'display': 'none'}
    visible = None  # let CSS class control the style

    new_capacity_only = 'new_only' in (new_capacity_checkbox or [])
    start_year, end_year = year_range if year_range else [2025, 2040]

    raw_df = _prepare_volume_country_dataframe(
        scenario=scenario,
        selected_unit='mtpa',
        new_capacity_only=new_capacity_only,
        start_year=start_year,
        end_year=end_year,
        breakdown='country',
    )
    available_countries = _get_available_volume_countries(raw_df)
    options = [{'label': country, 'value': country} for country in available_countries]

    if current_selection is None:
        selected_values = default_selected_countries(available_countries)
    else:
        selected_values = [
            country for country in current_selection if country in available_countries
        ]
        if current_selection and not selected_values:
            selected_values = default_selected_countries(available_countries)

    section_style = visible if is_country else hidden
    return options, selected_values, section_style, section_style


@callback(
    Output('capacity-volume-area-chart', 'figure'),
    Output('capacity-country-table-container', 'children'),
    Input('capacity-scenario-dropdown', 'value'),
    Input('capacity-unit-dropdown', 'value'),
    Input('capacity-new-capacity-checkbox', 'value'),
    Input('capacity-breakdown-dropdown', 'value'),
    Input('capacity-country-columns-dropdown', 'value'),
    Input('capacity-other-country-mode', 'value'),
    Input('capacity-year-range-slider', 'value'),
    Input('global-refresh-button', 'n_clicks'),
    prevent_initial_call=True
)
def update_capacity_section(
    scenario,
    selected_unit,
    new_capacity_checkbox,
    breakdown,
    selected_countries,
    other_countries_mode,
    year_range,
    n_clicks,
):
    """Update the production page chart and table from one shared dataset."""
    if not scenario or not selected_unit:
        raise PreventUpdate

    breakdown = breakdown or 'country'
    new_capacity_only = 'new_only' in (new_capacity_checkbox or [])
    start_year, end_year = year_range if year_range else [2025, 2040]

    raw_df = _prepare_volume_country_dataframe(
        scenario=scenario,
        selected_unit=selected_unit,
        new_capacity_only=new_capacity_only,
        start_year=start_year,
        end_year=end_year,
        breakdown=breakdown,
    )

    if raw_df.empty:
        empty_message = "No volume data available for the current selection."
        return _create_empty_volume_figure(empty_message), _create_empty_table_state(empty_message)

    if breakdown != 'country':
        matrix_df = _build_direct_pivot_matrix(raw_df, selected_unit)
        if matrix_df.empty:
            empty_message = "No volume data available for the current selection."
            return _create_empty_volume_figure(empty_message), html.Div()
        return create_volume_country_area_chart(matrix_df, selected_unit), html.Div()

    available_countries = _get_available_volume_countries(raw_df)
    if selected_countries is None:
        resolved_countries = default_selected_countries(available_countries)
    else:
        resolved_countries = [
            country for country in selected_countries if country in available_countries
        ]

    if resolved_countries == [] and other_countries_mode == 'exclude':
        empty_message = "Select at least one country or switch to Rest of the World mode."
        return _create_empty_volume_figure(empty_message), _create_empty_table_state(empty_message)

    matrix_df = _build_volume_country_matrix(
        raw_df,
        resolved_countries,
        other_countries_mode,
        selected_unit,
    )

    if matrix_df.empty:
        empty_message = "No volume data available for the current selection."
        return _create_empty_volume_figure(empty_message), _create_empty_table_state(empty_message)

    return (
        create_volume_country_area_chart(matrix_df, selected_unit),
        _create_volume_country_table('capacity-country-table', matrix_df),
    )


@callback(
    Output('capacity-download-excel', 'data'),
    Input('capacity-export-excel-button', 'n_clicks'),
    State('capacity-scenario-dropdown', 'value'),
    State('capacity-unit-dropdown', 'value'),
    State('capacity-new-capacity-checkbox', 'value'),
    State('capacity-breakdown-dropdown', 'value'),
    State('capacity-country-columns-dropdown', 'value'),
    State('capacity-other-country-mode', 'value'),
    State('capacity-year-range-slider', 'value'),
    prevent_initial_call=True
)
def export_capacity_to_excel(
    n_clicks,
    scenario,
    selected_unit,
    new_capacity_checkbox,
    breakdown,
    selected_countries,
    other_countries_mode,
    year_range,
):
    """Export the visible month-by-group matrix to Excel."""
    if n_clicks == 0:
        return None

    breakdown = breakdown or 'country'
    new_capacity_only = 'new_only' in (new_capacity_checkbox or [])
    start_year, end_year = year_range if year_range else [2025, 2040]

    raw_df = _prepare_volume_country_dataframe(
        scenario=scenario,
        selected_unit=selected_unit,
        new_capacity_only=new_capacity_only,
        start_year=start_year,
        end_year=end_year,
        breakdown=breakdown,
    )

    if raw_df.empty:
        return None

    if breakdown != 'country':
        export_df = _build_direct_pivot_matrix(raw_df, selected_unit)
    else:
        available_countries = _get_available_volume_countries(raw_df)
        if selected_countries is None:
            resolved_countries = default_selected_countries(available_countries)
        else:
            resolved_countries = [
                country for country in selected_countries if country in available_countries
            ]
        if resolved_countries == [] and other_countries_mode == 'exclude':
            return None
        export_df = _build_volume_country_matrix(
            raw_df,
            resolved_countries,
            other_countries_mode,
            selected_unit,
        )

    if export_df.empty:
        return None

    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        export_df.to_excel(writer, sheet_name='Volume Data', index=False)

        worksheet = writer.sheets['Volume Data']
        for idx, col in enumerate(export_df.columns):
            max_length = max(
                export_df[col].astype(str).apply(len).max(),
                len(str(col))
            ) + 2

            col_letter = ''
            temp_idx = idx + 1
            while temp_idx > 0:
                temp_idx -= 1
                col_letter = chr(65 + (temp_idx % 26)) + col_letter
                temp_idx //= 26

            worksheet.column_dimensions[col_letter].width = min(max_length, 50)

    output.seek(0)

    unit_label = selected_unit.upper()
    new_cap_label = '_NewCapacity' if new_capacity_only else ''
    breakdown_label = {'country': 'Country', 'project': 'Plant', 'train': 'Train'}[breakdown]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'LNG_Production_{breakdown_label}Matrix_{unit_label}{new_cap_label}_{timestamp}.xlsx'

    return dcc.send_bytes(output.getvalue(), filename)
