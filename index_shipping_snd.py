from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from dash import dcc, html
from dash.development.base_component import Component
from dash.dependencies import Input, Output, State

from app import app
import pages.capacity
import pages.contracts
import pages.country_mappings
import pages.demand
import pages.exporter_detail
import pages.exporters
import pages.fleet_metrics
import pages.importer_detail
import pages.importers
import pages.lng_phys_snapshot
import pages.market_balance
import pages.production
import pages.supply
import pages.terminal_adjustments
import pages.train_names_mapping


@dataclass(frozen=True)
class PageSpec:
    path: str
    title: str
    nav_id: str | None
    layout_factory: Callable[[], Component] | None
    aliases: tuple[str, ...] = ()
    redirect_to: str | None = None
    nav_label: str | None = None


def _static_layout(layout: Component) -> Callable[[], Component]:
    return lambda: layout


PAGE_SPECS = (
    PageSpec(
        path="/fleet_metrics",
        title="LNG Shipping - Fleet Metrics",
        nav_id="nav-fleet-metrics",
        nav_label="Fleet Metrics",
        layout_factory=_static_layout(pages.fleet_metrics.layout),
    ),
    PageSpec(
        path="/supply",
        title="LNG Shipping - Supply",
        nav_id="nav-supply",
        nav_label="Supply",
        layout_factory=_static_layout(pages.supply.layout),
    ),
    PageSpec(
        path="/demand",
        title="LNG Shipping - Demand",
        nav_id="nav-demand",
        nav_label="Demand",
        layout_factory=_static_layout(pages.demand.layout),
    ),
    PageSpec(
        path="/market_balance",
        title="LNG Shipping - Market Balance",
        nav_id="nav-market-balance",
        nav_label="Market Balance",
        layout_factory=_static_layout(pages.market_balance.layout),
    ),
    PageSpec(
        path="/lng-phys-snapshot",
        title="LNG Shipping - LNG Physical Snapshot",
        nav_id="nav-lng-phys-snapshot",
        nav_label="LNG Phys Snapshot",
        layout_factory=_static_layout(pages.lng_phys_snapshot.layout),
    ),
    PageSpec(
        path="/exporters",
        title="LNG Shipping - Exporters",
        nav_id="nav-exporters",
        nav_label="Exporters",
        layout_factory=_static_layout(pages.exporters.layout),
        aliases=("/",),
    ),
    PageSpec(
        path="/importers",
        title="LNG Shipping - Importers",
        nav_id="nav-importers",
        nav_label="Importers",
        layout_factory=_static_layout(pages.importers.layout),
    ),
    PageSpec(
        path="/exporter_detail",
        title="LNG Shipping - Exporter Detail",
        nav_id="nav-exporter-detail",
        nav_label="Exporter Detail",
        layout_factory=_static_layout(pages.exporter_detail.layout),
    ),
    PageSpec(
        path="/importer_detail",
        title="LNG Shipping - Importer Detail",
        nav_id="nav-importer-detail",
        nav_label="Importer Detail",
        layout_factory=_static_layout(pages.importer_detail.layout),
    ),
    PageSpec(
        path="/contracts",
        title="LNG Shipping - Contracts",
        nav_id="nav-contracts",
        nav_label="Contracts",
        layout_factory=pages.contracts.layout,
    ),
    PageSpec(
        path="/production",
        title="LNG Shipping - Production",
        nav_id="nav-production",
        nav_label="Production",
        layout_factory=pages.production.layout,
    ),
    PageSpec(
        path="/capacity",
        title="LNG Shipping - Capacity",
        nav_id="nav-capacity",
        nav_label="Capacity",
        layout_factory=_static_layout(pages.capacity.layout),
    ),
    PageSpec(
        path="/mappings",
        title="LNG Shipping - Country Mappings",
        nav_id="nav-mappings",
        nav_label="Mappings",
        layout_factory=_static_layout(pages.country_mappings.layout),
        aliases=("/country_mappings",),
    ),
    PageSpec(
        path="/train_names_mapping",
        title="LNG Shipping - Train Mapping",
        nav_id="nav-mappings",
        layout_factory=_static_layout(pages.train_names_mapping.layout),
    ),
    PageSpec(
        path="/terminal_adjustments",
        title="LNG Shipping - Terminal Adjustments",
        nav_id=None,
        layout_factory=pages.terminal_adjustments.layout,
    ),
    PageSpec(
        path="/balance",
        title="LNG Shipping - Supply",
        nav_id="nav-supply",
        layout_factory=None,
        redirect_to="/supply",
    ),
    PageSpec(
        path="/terminals",
        title="LNG Shipping - Capacity",
        nav_id="nav-capacity",
        layout_factory=None,
        redirect_to="/capacity",
    ),
)

PAGE_SPEC_BY_PATH = {
    route_path: page_spec
    for page_spec in PAGE_SPECS
    for route_path in (page_spec.path, *page_spec.aliases)
}

CLIENT_PAGE_REGISTRY = {
    route_path: {
        "title": page_spec.title,
        "navId": page_spec.nav_id or "",
    }
    for route_path, page_spec in PAGE_SPEC_BY_PATH.items()
}


def _build_routed_page(page_spec: PageSpec) -> Component:
    page_heading = page_spec.title.removeprefix("LNG Shipping - ")
    return html.Main(
        [
            html.H1(page_heading, className="visually-hidden-page-title"),
            page_spec.layout_factory(),
        ],
        className="routed-page-main",
    )


_EXPORTERS_ROUTED_PAGE = _build_routed_page(
    PAGE_SPEC_BY_PATH["/exporters"]
)


nav_links = html.Header(
    [
        html.Div(
            [
                html.Nav(
                    [
                        html.Div(
                            [
                                dcc.Link(
                                    page_spec.nav_label,
                                    href=page_spec.path,
                                    id=page_spec.nav_id,
                                    className="nav-link-secondary",
                                )
                                for page_spec in PAGE_SPECS
                                if page_spec.nav_label is not None
                            ],
                            className="nav-group-secondary",
                        )
                    ],
                    className="main-navigation",
                    **{"aria-label": "Dashboard pages"},
                ),
                html.Div(
                    [
                        html.Button(
                            "Refresh Data",
                            id="global-refresh-button",
                            n_clicks=0,
                            className="btn-refresh",
                        ),
                    ],
                    className="top-bar-controls",
                ),
            ],
            className="top-bar-content",
        )
    ],
    className="top-bar-header",
)

app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        dcc.Store(id="page-registry", data=CLIENT_PAGE_REGISTRY),
        nav_links,
        html.Div(id="page-content"),
    ]
)


@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
)
def display_page(pathname):
    page_spec = PAGE_SPEC_BY_PATH.get(pathname)
    if page_spec is None:
        return "404 - Page not found"

    if page_spec.redirect_to:
        redirect_id = (
            "redirect-supply-from-balance"
            if pathname == "/balance"
            else "redirect-capacity-from-terminals"
        )
        return dcc.Location(pathname=page_spec.redirect_to, id=redirect_id)

    if page_spec.path == "/exporters":
        return _EXPORTERS_ROUTED_PAGE
    return _build_routed_page(page_spec)


app.clientside_callback(
    """
    function(pathname, pageRegistry) {
        const pageSpec = (pageRegistry || {})[pathname];
        document.title = pageSpec
            ? pageSpec.title
            : 'LNG Shipping - Page Not Found';

        const navLinks = document.querySelectorAll(
            '.nav-link-primary, .nav-link-secondary'
        );
        navLinks.forEach(link => link.classList.remove('active'));

        if (pageSpec && pageSpec.navId) {
            const activeLink = document.getElementById(pageSpec.navId);
            if (activeLink) {
                activeLink.classList.add('active');
            }
        }

        return {};
    }
    """,
    Output("page-content", "style"),
    Input("url", "pathname"),
    State("page-registry", "data"),
)


if __name__ == "__main__":
    app.run(debug=True, port=8067)
