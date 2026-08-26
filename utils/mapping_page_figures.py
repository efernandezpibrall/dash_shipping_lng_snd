"""Shared helpers for plant/train mapping maintenance pages."""

from dash import html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd

from utils.ag_grid_tables import create_ag_grid_from_datatable
from utils.table_styles import StandardTableStyleManager


def clean_text_value(value):
    if pd.isna(value):
        return ""

    text_value = str(value).strip()
    if not text_value:
        return ""

    return " ".join(text_value.split())


def build_dropdown_options(series: pd.Series) -> list[dict[str, object]]:
    return [
        {"label": value, "value": value}
        for value in sorted(series.replace("", pd.NA).dropna().unique())
    ]


def build_summary_card_row(card_specs):
    cards = []
    for label, value in card_specs:
        cards.append(
            dbc.Col(
                [
                    dbc.Card(
                        [
                            dbc.CardBody(
                                [
                                    html.H6(
                                        label,
                                        className="text-secondary",
                                        style={"marginBottom": "8px"},
                                    ),
                                    html.H3(value, className="text-primary font-bold"),
                                ]
                            )
                        ],
                        className="shadow-sm h-100",
                    )
                ],
                width=3,
            )
        )

    return dbc.Row(cards, className="mb-4")


def build_mapping_table(
    *,
    table_id,
    display_columns,
    editable_columns,
    style_data_conditional,
    style_cell_conditional,
    numeric_columns=None,
):
    table_config = StandardTableStyleManager.get_base_datatable_config()
    numeric_columns = set(numeric_columns or [])

    columns = []
    for column_name in display_columns:
        column_config = {
            "name": column_name.replace("_", " ").title(),
            "id": column_name,
            "editable": column_name in editable_columns,
        }
        if column_name in numeric_columns:
            column_config["type"] = "numeric"
        columns.append(column_config)

    return create_ag_grid_from_datatable(
        id=table_id,
        columns=columns,
        data=[],
        editable=True,
        sort_action="native",
        filter_action="native",
        page_action="native",
        page_size=50,
        export_format="xlsx",
        style_table=table_config["style_table"],
        style_data_conditional=table_config["style_data_conditional"] + list(style_data_conditional or []),
        style_cell_conditional=style_cell_conditional or [],
    )


def filter_mapping_dataframe(
    df: pd.DataFrame,
    filters,
    search_columns,
    search_text,
) -> pd.DataFrame:
    filtered_df = df.copy()

    for column_name, selected_values in filters:
        if selected_values:
            filtered_df = filtered_df[filtered_df[column_name].isin(selected_values)]

    search_value = clean_text_value(search_text).lower()
    if search_value:
        combined_search = (
            filtered_df[search_columns]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
            .str.lower()
        )
        filtered_df = filtered_df[combined_search.str.contains(search_value, regex=False)]

    return filtered_df.reset_index(drop=True)


def build_empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 16, "color": "#64748b"},
    )
    fig.update_layout(
        template="plotly_white",
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return fig


def build_bar_figure(series: pd.Series, title: str, color: str) -> go.Figure:
    import plotly.express as px

    if series.empty:
        return build_empty_figure("No data available for the current selection.")

    plot_df = series.reset_index()
    plot_df.columns = ["label", "count"]

    fig = px.bar(
        plot_df,
        x="count",
        y="label",
        orientation="h",
        text="count",
    )
    fig.update_traces(marker_color=color, textposition="outside", cliponaxis=False)
    fig.update_layout(
        template="plotly_white",
        title=title,
        height=max(320, 60 + (len(plot_df) * 28)),
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
        xaxis_title="Rows",
        yaxis_title="",
        showlegend=False,
    )
    fig.update_yaxes(categoryorder="total ascending")
    return fig
