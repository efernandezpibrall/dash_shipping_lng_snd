from dash import dcc, html


MAPPINGS_NAV_ITEMS = [
    {
        "title": "Country Mapping",
        "href": "/mappings",
        "eyebrow": "Geography",
        "description": "Review country, basin, classification, and shipping-region mappings.",
    },
    {
        "title": "Provider Train Links",
        "href": "/train_names_mapping",
        "eyebrow": "Canonical Trains",
        "description": "Allocate provider source records to Capacity business trains.",
    },
]


def create_mappings_section_header(title, description, active_href):
    nav_links = []
    for item in MAPPINGS_NAV_ITEMS:
        class_name = "mapping-nav-link active" if item["href"] == active_href else "mapping-nav-link"
        nav_links.append(
            dcc.Link(
                [
                    html.Span(item["eyebrow"], className="mapping-nav-kicker"),
                    html.Span(item["title"], className="mapping-nav-title"),
                    html.Span(item["description"], className="mapping-nav-description"),
                ],
                href=item["href"],
                className=class_name,
            )
        )

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Span("Mappings Workspace", className="mappings-section-kicker"),
                            html.H2(title, className="page-title", style={"marginBottom": "10px"}),
                            html.P(description, className="mappings-section-description"),
                        ],
                        className="mappings-section-copy",
                    ),
                    html.Div(nav_links, className="mappings-nav-grid"),
                ],
                className="mappings-section-header",
            )
        ],
        className="mappings-section-shell",
    )
