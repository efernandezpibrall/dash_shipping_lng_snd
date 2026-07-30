import json
from pathlib import Path
import subprocess
import sys
import textwrap


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_fresh_app_probe():
    probe = textwrap.dedent(
        """
        import json
        import sys

        import index_shipping_snd as index
        from dash._callback import GLOBAL_CALLBACK_MAP


        def walk(component):
            if component is None:
                return
            if isinstance(component, (list, tuple)):
                for child in component:
                    yield from walk(child)
                return
            yield component
            yield from walk(getattr(component, "children", None))


        shipping_imported_by_active_app = "pages.shipping_balance" in sys.modules
        index.app._setup_server()
        active_callback_keys = set(index.app.callback_map)
        active_callback_count = len(active_callback_keys)

        nav_items = [
            {
                "id": getattr(component, "id", None),
                "href": getattr(component, "href", None),
            }
            for component in walk(index.nav_links)
            if getattr(component, "id", None)
        ]

        global_before_shipping_import = set(GLOBAL_CALLBACK_MAP)
        import pages.shipping_balance  # noqa: F401
        shipping_callback_keys = (
            set(GLOBAL_CALLBACK_MAP) - global_before_shipping_import
        )

        shipping_markers = (
            "shipping-balance",
            "global-shipping-balance",
            "fleet-stats-chart",
            "intracountry-",
            "demand-regional-",
            "supply-regional-",
        )
        active_shipping_callbacks = sorted(
            key
            for key in active_callback_keys
            if any(marker in key for marker in shipping_markers)
        )

        print(
            "APP_PROBE="
            + json.dumps(
                {
                    "shipping_imported_by_active_app": shipping_imported_by_active_app,
                    "active_callback_count": active_callback_count,
                    "active_callback_count_after_shipping_import": len(index.app.callback_map),
                    "shipping_callback_count_when_explicitly_imported": len(
                        shipping_callback_keys
                    ),
                    "active_shipping_callbacks": active_shipping_callbacks,
                    "nav_items": nav_items,
                    "root_and_exporters_are_same_component": (
                        index.display_page("/") is index.display_page("/exporters")
                    ),
                    "root_client_page": index.CLIENT_PAGE_REGISTRY["/"],
                    "exporters_client_page": (
                        index.CLIENT_PAGE_REGISTRY["/exporters"]
                    ),
                    "shipping_balance_route": index.display_page("/shipping_balance"),
                    "app_title": index.app.title,
                }
            )
        )
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    output_line = next(
        line for line in completed.stdout.splitlines() if line.startswith("APP_PROBE=")
    )
    return json.loads(output_line.removeprefix("APP_PROBE="))


def test_shipping_balance_is_retained_but_not_registered_in_active_app():
    probe = _run_fresh_app_probe()

    assert probe["shipping_imported_by_active_app"] is False
    assert probe["shipping_callback_count_when_explicitly_imported"] == 19
    assert probe["active_shipping_callbacks"] == []
    assert (
        probe["active_callback_count_after_shipping_import"]
        == probe["active_callback_count"]
    )


def test_exporters_is_the_landing_page_and_shipping_balance_is_not_navigable():
    probe = _run_fresh_app_probe()
    nav_by_id = {item["id"]: item["href"] for item in probe["nav_items"]}

    assert probe["root_and_exporters_are_same_component"] is True
    assert probe["shipping_balance_route"] == "404 - Page not found"
    assert probe["app_title"] == "LNG Shipping - Exporters"
    assert nav_by_id["nav-exporters"] == "/exporters"
    assert "nav-shipping-balance" not in nav_by_id
    assert "/shipping_balance" not in nav_by_id.values()


def test_clientside_title_and_active_navigation_use_exporters_for_root():
    route_source = (REPO_ROOT / "index_shipping_snd.py").read_text()
    probe = _run_fresh_app_probe()

    assert probe["root_client_page"] == probe["exporters_client_page"] == {
        "title": "LNG Shipping - Exporters",
        "navId": "nav-exporters",
    }
    assert "CLIENT_PAGE_REGISTRY" in route_source
    assert "pageSpec.title" in route_source
    assert "pageSpec.navId" in route_source
    assert "import pages.shipping_balance" not in route_source
    assert "nav-shipping-balance" not in route_source
    assert "/shipping_balance" not in route_source
