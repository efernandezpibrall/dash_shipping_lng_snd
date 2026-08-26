# app.py
import logging
import os

from dash import Dash
import dash_bootstrap_components as dbc
from flask import request
from sqlalchemy import text


LOG_LEVEL_NAME = os.getenv("DASH_LOG_LEVEL", "WARNING").upper()
LOG_LEVEL = getattr(logging, LOG_LEVEL_NAME, logging.WARNING)
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logging.getLogger().setLevel(LOG_LEVEL)


def _environment_flag(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().casefold() not in {
        "",
        "0",
        "false",
        "no",
        "off",
    }


PLOTLY_BASIC_ENABLED = _environment_flag(
    "DASH_PLOTLY_BASIC_ENABLED",
    default=True,
)
HTTP_COMPRESSION_ENABLED = _environment_flag(
    "DASH_HTTP_COMPRESSION_ENABLED",
    default=True,
)

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    assets_ignore=(
        None
        if PLOTLY_BASIC_ENABLED
        else r"^00_plotly-basic-3\.1\.0\.min\.js$"
    ),
    compress=HTTP_COMPRESSION_ENABLED,
)
app.title = "LNG Shipping - Exporters"
server = app.server


@server.after_request
def optimize_static_asset_delivery(response):
    """Compress and cache Dash assets whose URL carries a version token."""

    if request.path.startswith("/assets/") and response.status_code == 200:
        if HTTP_COMPRESSION_ENABLED:
            response.direct_passthrough = False
            response.set_data(response.get_data())
        if "m" in request.args:
            response.headers["Cache-Control"] = (
                "public, max-age=31536000, immutable"
            )
    return response


@server.get("/health")
def health():
    """Process liveness endpoint for the local service manager."""

    return {
        "service": "dash_shipping_lng_snd",
        "status": "ok",
    }


@server.get("/ready")
def ready():
    """Readiness endpoint proving the configured database is reachable."""

    try:
        from utils.database import get_database_engine

        with get_database_engine().connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception:
        logging.getLogger(__name__).exception(
            "Dashboard readiness database check failed"
        )
        return {
            "service": "dash_shipping_lng_snd",
            "status": "unavailable",
        }, 503
    return {
        "database": "reachable",
        "service": "dash_shipping_lng_snd",
        "status": "ready",
    }
