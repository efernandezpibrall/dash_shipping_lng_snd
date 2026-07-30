from dash import dcc, html

import index_shipping_snd as index


EXPECTED_ROUTES = {
    "/",
    "/balance",
    "/capacity",
    "/contracts",
    "/country_mappings",
    "/demand",
    "/exporter_detail",
    "/exporters",
    "/fleet_metrics",
    "/importer_detail",
    "/importers",
    "/lng-phys-snapshot",
    "/mappings",
    "/market_balance",
    "/production",
    "/supply",
    "/terminal_adjustments",
    "/terminals",
    "/train_names_mapping",
}


def test_page_registry_preserves_every_route_and_redirect():
    assert set(index.PAGE_SPEC_BY_PATH) == EXPECTED_ROUTES
    assert index.PAGE_SPEC_BY_PATH["/"].path == "/exporters"
    assert index.PAGE_SPEC_BY_PATH["/country_mappings"].path == "/mappings"
    assert index.PAGE_SPEC_BY_PATH["/balance"].redirect_to == "/supply"
    assert index.PAGE_SPEC_BY_PATH["/terminals"].redirect_to == "/capacity"


def test_client_registry_preserves_titles_and_active_nav_ids():
    assert index.CLIENT_PAGE_REGISTRY["/"]["title"] == (
        "LNG Shipping - Exporters"
    )
    assert index.CLIENT_PAGE_REGISTRY["/balance"]["navId"] == "nav-supply"
    assert index.CLIENT_PAGE_REGISTRY["/terminals"]["navId"] == (
        "nav-capacity"
    )
    assert index.CLIENT_PAGE_REGISTRY["/train_names_mapping"]["navId"] == (
        "nav-mappings"
    )


def test_router_wraps_page_with_hidden_heading_and_preserves_redirect_ids():
    routed = index.display_page("/supply")
    assert isinstance(routed, html.Main)
    assert isinstance(routed.children[0], html.H1)
    assert routed.children[0].children == "Supply"
    assert routed.children[0].className == "visually-hidden-page-title"

    balance_redirect = index.display_page("/balance")
    terminals_redirect = index.display_page("/terminals")
    assert isinstance(balance_redirect, dcc.Location)
    assert balance_redirect.pathname == "/supply"
    assert balance_redirect.id == "redirect-supply-from-balance"
    assert terminals_redirect.pathname == "/capacity"
    assert terminals_redirect.id == "redirect-capacity-from-terminals"


def test_registry_has_exact_navigation_order_and_ids():
    navigation_specs = [
        spec
        for spec in index.PAGE_SPECS
        if spec.nav_label is not None
    ]
    assert [spec.nav_label for spec in navigation_specs] == [
        "Fleet Metrics",
        "Supply",
        "Demand",
        "Market Balance",
        "LNG Phys Snapshot",
        "Exporters",
        "Importers",
        "Exporter Detail",
        "Importer Detail",
        "Contracts",
        "Production",
        "Capacity",
        "Mappings",
    ]
    assert len({spec.nav_id for spec in navigation_specs}) == 13
