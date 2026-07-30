# app.py
import os

from dash import Dash
import dash_bootstrap_components as dbc


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
