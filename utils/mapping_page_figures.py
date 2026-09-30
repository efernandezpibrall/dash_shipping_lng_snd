"""Shared helpers for plant/train mapping maintenance pages."""

from dash import html
import dash_bootstrap_components as dbc
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
