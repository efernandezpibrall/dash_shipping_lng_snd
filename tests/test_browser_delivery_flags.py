import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import textwrap


REPO_ROOT = Path(__file__).resolve().parents[1]
PLOTLY_BASIC_ASSET = (
    REPO_ROOT / "assets" / "00_plotly-basic-3.1.0.min.js"
)
PLOTLY_BASIC_SHA256 = (
    "80d756297ee4b5e701cee4e5c88aa52ec822c3ed84fd23bda6e20b03d10492a6"
)


def _fresh_delivery_probe(*, plotly_basic, compression):
    script = textwrap.dedent(
        """
        import gzip
        import hashlib
        import json
        import re

        import index_shipping_snd as index

        client = index.app.server.test_client()
        html = client.get("/").get_data(as_text=True)
        script_sources = re.findall(r'<script[^>]+src="([^"]+)"', html)
        paths = ["/", "/_dash-layout", "/_dash-dependencies"]
        responses = {}
        for path in paths:
            identity = client.get(path, headers={"Accept-Encoding": "identity"})
            encoded = client.get(path, headers={"Accept-Encoding": "gzip"})
            decoded = (
                gzip.decompress(encoded.data)
                if encoded.headers.get("Content-Encoding") == "gzip"
                else encoded.data
            )
            identity_bytes = identity.data
            decoded_bytes = decoded
            if path == "/":
                identity_bytes = re.sub(
                    rb'"end_id":"[^"]+"',
                    b'"end_id":"<request-token>"',
                    identity_bytes,
                )
                decoded_bytes = re.sub(
                    rb'"end_id":"[^"]+"',
                    b'"end_id":"<request-token>"',
                    decoded_bytes,
                )
            responses[path] = {
                "identity_size": len(identity.data),
                "encoded_size": len(encoded.data),
                "content_encoding": encoded.headers.get("Content-Encoding"),
                "vary": encoded.headers.get("Vary"),
                "same_content": (
                    hashlib.sha256(identity_bytes).hexdigest()
                    == hashlib.sha256(decoded_bytes).hexdigest()
                ),
            }
        print(
            "DELIVERY_PROBE="
            + json.dumps(
                {
                    "script_sources": script_sources,
                    "responses": responses,
                }
            )
        )
        """
    )
    environment = os.environ.copy()
    environment["DASH_PLOTLY_BASIC_ENABLED"] = (
        "1" if plotly_basic else "0"
    )
    environment["DASH_HTTP_COMPRESSION_ENABLED"] = (
        "1" if compression else "0"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    output = next(
        line
        for line in completed.stdout.splitlines()
        if line.startswith("DELIVERY_PROBE=")
    )
    return json.loads(output.removeprefix("DELIVERY_PROBE="))


def test_plotly_basic_asset_checksum_and_static_trace_inventory():
    assert hashlib.sha256(PLOTLY_BASIC_ASSET.read_bytes()).hexdigest() == (
        PLOTLY_BASIC_SHA256
    )

    source = "\n".join(
        path.read_text()
        for root in (REPO_ROOT / "pages", REPO_ROOT / "utils")
        for path in root.rglob("*.py")
    )
    trace_constructors = set(
        re.findall(r"\b(?:go|px)\.[A-Za-z0-9_]+", source)
    )
    assert trace_constructors == {
        "go.Bar",
        "go.Figure",
        "go.Scatter",
        "px.area",
        "px.bar",
        "px.line",
    }


def test_plotly_basic_rollback_switch_changes_fresh_index_scripts():
    enabled = _fresh_delivery_probe(
        plotly_basic=True,
        compression=False,
    )
    disabled = _fresh_delivery_probe(
        plotly_basic=False,
        compression=False,
    )
    asset_name = "00_plotly-basic-3.1.0.min.js"

    enabled_matches = [
        source
        for source in enabled["script_sources"]
        if asset_name in source
    ]
    disabled_matches = [
        source
        for source in disabled["script_sources"]
        if asset_name in source
    ]
    assert len(enabled_matches) == 1
    assert disabled_matches == []
    assert not any(
        "plotly.min.js" in source
        for source in enabled["script_sources"]
    )


def test_http_compression_flag_preserves_bytes_and_clears_ten_percent():
    enabled = _fresh_delivery_probe(
        plotly_basic=False,
        compression=True,
    )
    disabled = _fresh_delivery_probe(
        plotly_basic=False,
        compression=False,
    )

    for path, response in enabled["responses"].items():
        assert response["content_encoding"] == "gzip"
        assert response["vary"] == "Accept-Encoding"
        assert response["same_content"] is True
        assert response["encoded_size"] < response["identity_size"] * 0.9
        assert disabled["responses"][path]["content_encoding"] is None
        assert disabled["responses"][path]["same_content"] is True
