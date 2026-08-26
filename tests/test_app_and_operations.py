# Consolidated from test_app_health.py.

from app import health, ready, server


def test_health_is_real_json_endpoint():
    routes = {rule.rule for rule in server.url_map.iter_rules()}
    payload = health()

    assert "/health" in routes
    assert payload == {
        "service": "dash_shipping_lng_snd",
        "status": "ok",
    }


def test_ready_reports_database_success(monkeypatch):
    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, _query):
            return None

    class Engine:
        def connect(self):
            return Connection()

    monkeypatch.setattr(
        "utils.database.get_database_engine",
        lambda: Engine(),
    )

    payload = ready()

    assert payload["status"] == "ready"


def test_ready_reports_database_failure_without_leaking_exception(monkeypatch):
    monkeypatch.setattr(
        "utils.database.get_database_engine",
        lambda: (_ for _ in ()).throw(RuntimeError("secret detail")),
    )

    payload, status_code = ready()

    assert status_code == 503
    assert payload == {
        "service": "dash_shipping_lng_snd",
        "status": "unavailable",
    }
    assert "secret detail" not in str(payload)


# Consolidated from test_browser_delivery_flags.py.

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import textwrap


_DELIVERY_REPO_ROOT = Path(__file__).resolve().parents[1]
PLOTLY_BASIC_ASSET = (
    _DELIVERY_REPO_ROOT / "assets" / "00_plotly-basic-3.1.0.min.js"
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
        paths = [
            "/",
            "/_dash-layout",
            "/_dash-dependencies",
            "/assets/00_plotly-basic-3.1.0.min.js?m=test",
            "/assets/styles.css?m=test",
            "/assets/balance_scroll_sync.js?m=test",
        ]
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
                "cache_control": encoded.headers.get("Cache-Control"),
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
        cwd=_DELIVERY_REPO_ROOT,
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
        for root in (_DELIVERY_REPO_ROOT / "pages", _DELIVERY_REPO_ROOT / "utils")
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
        if path.startswith("/assets/"):
            assert response["cache_control"] == (
                "public, max-age=31536000, immutable"
            )
            assert disabled["responses"][path]["cache_control"] == (
                "public, max-age=31536000, immutable"
            )


# Consolidated from test_capacity_performance.py.

import json
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pandas as pd
import pytest
from dash.exceptions import PreventUpdate

from pages import capacity


def _woodmac_monthly_fixture() -> pd.DataFrame:
    rows = []
    for month, first_capacity, second_capacity in [
        ("2025-12-01", 0.0, 1.0),
        ("2026-01-01", 2.0, 1.0),
        ("2026-02-01", 2.0, 0.0),
    ]:
        for train_id, train_label, raw_name, value in [
            (1, 1, "Alpha Train 1", first_capacity),
            (2, 2, "Alpha Train 2", second_capacity),
        ]:
            rows.append(
                {
                    "month": pd.Timestamp(month),
                    "country_name": "Country A",
                    "plant_name": "Alpha",
                    "raw_plant_name": "Alpha Raw",
                    "plant_mapping_applied": True,
                    "raw_train_name": raw_name,
                    "raw_train_display_name": raw_name,
                    "woodmac_fid_date": "2025-01-01",
                    "train": train_label,
                    "allocation_share": 1.0,
                    "train_mapping_applied": True,
                    "lng_train_name": raw_name,
                    "lng_train_name_short": raw_name,
                    "id_plant": 10,
                    "id_lng_train": train_id,
                    "capacity_mtpa": value,
                }
            )
    return pd.DataFrame(rows)


def test_woodmac_canonical_keys_keep_gl1z_and_gl2z_separate():
    raw_df = pd.DataFrame(
        [
            {
                "month": pd.Timestamp("2026-01-01"),
                "country_name": "Algeria",
                "plant_name": "Algeria LNG (Bethioua)",
                "raw_plant_name": "Algeria LNG (Bethioua)",
                "raw_train_name": "Bethioua Train 1",
                "raw_train_display_name": "Algeria LNG (Bethioua) Train 1 (GL1Z)",
                "woodmac_fid_date": "",
                "train": 1,
                "allocation_share": 1.0,
                "train_mapping_applied": False,
                "id_plant": 151,
                "id_lng_train": 8,
                "capacity_mtpa": 1.4,
                "terminal_key": "terminal-bethioua",
                "train_key": "train-gl1z-1",
                "canonical_terminal_name": "Algeria LNG (Bethioua)",
                "canonical_train_label": "Train 1 (GL1Z)",
            },
            {
                "month": pd.Timestamp("2026-01-01"),
                "country_name": "Algeria",
                "plant_name": "Algeria LNG (Bethioua)",
                "raw_plant_name": "Algeria LNG (Bethioua)",
                "plant_mapping_applied": False,
                "raw_train_name": "Bethioua Train 1",
                "raw_train_display_name": "Algeria LNG (Bethioua) Train 1 (GL2Z)",
                "woodmac_fid_date": "",
                "train": 1,
                "allocation_share": 1.0,
                "train_mapping_applied": False,
                "id_plant": 151,
                "id_lng_train": 14,
                "capacity_mtpa": 1.35,
                "terminal_key": "terminal-bethioua",
                "train_key": "train-gl2z-1",
                "canonical_terminal_name": "Algeria LNG (Bethioua)",
                "canonical_train_label": "Train 1 (GL2Z)",
            },
        ]
    )

    events = capacity._build_train_change_log(
        raw_df, None, "rest_of_world", None, None
    )
    scenario_rows = capacity._build_provider_scenario_rows_from_change_log(
        events, "woodmac"
    )

    assert events["series_key"].nunique() == 2
    assert scenario_rows["train_key"].tolist() == ["train-gl1z-1", "train-gl2z-1"]
    assert scenario_rows["train_label"].tolist() == [
        "Train 1 (GL1Z)",
        "Train 1 (GL2Z)",
    ]
    assert scenario_rows["scenario_row_key"].nunique() == 2


def test_woodmac_components_with_one_canonical_key_still_aggregate():
    raw_df = pd.DataFrame(
        [
            {
                "month": pd.Timestamp("2030-01-01"),
                "country_name": "United States",
                "plant_name": "Delfin FLNG",
                "raw_plant_name": "Delfin component",
                "plant_mapping_applied": True,
                "raw_train_name": f"Component {component}",
                "raw_train_display_name": f"Delfin component {component}",
                "woodmac_fid_date": "",
                "train": 1,
                "allocation_share": 1.0,
                "train_mapping_applied": True,
                "id_plant": provider_plant,
                "id_lng_train": provider_train,
                "capacity_mtpa": 2.2,
                "terminal_key": "terminal-delfin",
                "train_key": "train-delfin-1",
                "canonical_terminal_name": "Delfin FLNG",
                "canonical_train_label": "Train 1",
            }
            for component, provider_plant, provider_train in [
                ("A", 744, 375),
                ("B", 744, 515),
            ]
        ]
    )

    events = capacity._build_train_change_log(
        raw_df, None, "rest_of_world", None, None
    )

    assert events["series_key"].nunique() == 1
    assert events.iloc[0]["train_key"] == "train-delfin-1"
    assert events.iloc[0]["Delta MTPA"] == 4.4


def test_capacity_change_comparison_aligns_aliases_by_canonical_train_key():
    train_key = "c9c1cc87-081e-4a9a-8707-00e678257764"
    terminal_key = "terminal-golden-pass"
    woodmac_df = pd.DataFrame(
        [
            {
                "terminal_key": terminal_key,
                "train_key": train_key,
                "Canonical Train Label": "Train 1",
                "Effective Date": "2026-04-01",
                "Country": "United States",
                "Plant": "Golden Pass Export",
                "Train": 1,
                "Delta MTPA": 6.03,
                "Source Field": "plant_name",
                "Source Name": "Golden Pass Export",
                "Train Source Name": "Golden Pass Train 1",
                "Mapping Applied": True,
                "Train Mapping Applied": True,
            }
        ]
    )
    ea_df = pd.DataFrame(
        [
            {
                "terminal_key": terminal_key,
                "train_key": train_key,
                "Canonical Train Label": "Train 1",
                "Effective Date": "2026-04-01",
                "Country": "United States",
                "Plant": "Golden Pass Export",
                "Train": 1,
                "project_name": "Golden Pass Export",
                "train_name": "Train 1",
                "EA Adds (MTPA)": 6.0,
                "EA Reductions (MTPA)": 0.0,
                "EA Net Delta (MTPA)": 6.0,
                "Source Field": "project_name",
                "Source Name": "Golden Pass Export",
                "Train Source Name": "Train 1",
                "Mapping Applied": True,
                "Train Mapping Applied": True,
            }
        ]
    )
    scenario_rows_df = pd.DataFrame(
        [
            {
                "scenario_row_key": "golden-pass-1",
                "terminal_key": terminal_key,
                "train_key": train_key,
                "country_name": "United States",
                "plant_name": "Golden Pass",
                "train_label": "1",
                "scenario_first_date": pd.Timestamp("2026-04-01"),
                "scenario_capacity_mtpa": 6.03,
            }
        ]
    )
    internal_df = capacity._build_internal_scenario_change_log(
        scenario_rows_df,
        ["United States"],
        "exclude",
        "2026-04-01",
        "2026-04-01",
    )

    comparison_df = capacity._build_train_change_hierarchical_rows(
        woodmac_df,
        ea_df,
        internal_change_df=internal_df,
        time_view="monthly",
        detail_view="plants_trains",
    )

    train_rows = comparison_df[comparison_df["Type"].eq("train")]
    assert len(train_rows) == 1
    row = train_rows.iloc[0]
    assert row["Plant"] == "Golden Pass"
    assert row["Train"] == "1"
    assert row["Woodmac Net Delta (MTPA)"] == 6.03
    assert row["EA Net Delta (MTPA)"] == 6.0
    assert row[capacity.INTERNAL_SCENARIO_NET_COLUMN] == 6.03
    assert row["Woodmac Original Plant"] == "Golden Pass Export"


def test_capacity_change_comparison_keeps_same_numbered_canonical_trains_separate():
    terminal_key = "terminal-bethioua"
    woodmac_rows = []
    scenario_rows = []
    for suffix, train_key, capacity_value in [
        ("GL1Z", "train-gl1z-1", 1.4),
        ("GL2Z", "train-gl2z-1", 1.35),
    ]:
        woodmac_rows.append(
            {
                "terminal_key": terminal_key,
                "train_key": train_key,
                "Canonical Train Label": f"Train 1 ({suffix})",
                "Effective Date": "2026-01-01",
                "Country": "Algeria",
                "Plant": "Algeria LNG (Bethioua)",
                "Train": 1,
                "Delta MTPA": capacity_value,
                "Source Field": "plant_name",
                "Source Name": "Algeria LNG (Bethioua)",
                "Train Source Name": f"Bethioua Train 1 ({suffix})",
                "Mapping Applied": True,
                "Train Mapping Applied": True,
            }
        )
        scenario_rows.append(
            {
                "scenario_row_key": f"bethioua-{suffix}",
                "terminal_key": terminal_key,
                "train_key": train_key,
                "country_name": "Algeria",
                "plant_name": "Algeria LNG (Bethioua)",
                "train_label": f"Train 1 ({suffix})",
                "scenario_first_date": pd.Timestamp("2026-01-01"),
                "scenario_capacity_mtpa": capacity_value,
            }
        )

    internal_df = capacity._build_internal_scenario_change_log(
        pd.DataFrame(scenario_rows),
        ["Algeria"],
        "exclude",
        "2026-01-01",
        "2026-01-01",
    )
    comparison_df = capacity._build_train_change_hierarchical_rows(
        pd.DataFrame(woodmac_rows),
        pd.DataFrame(),
        internal_change_df=internal_df,
        detail_view="plants_trains",
    )

    train_rows = comparison_df[comparison_df["Type"].eq("train")]
    assert len(train_rows) == 2
    assert set(train_rows["Train"]) == {"Train 1 (GL1Z)", "Train 1 (GL2Z)"}
    assert sorted(train_rows["Woodmac Net Delta (MTPA)"].tolist()) == [1.35, 1.4]


def test_full_ea_change_log_can_be_losslessly_filtered_for_an_interaction():
    raw_df = pd.DataFrame(
        {
            "month": pd.to_datetime(["2026-01-01", "2026-03-01", "2026-03-01"]),
            "country_name": ["Country A", "Country A", "Country B"],
            "plant_name": ["Alpha", "Alpha", "Beta"],
            "project_name": ["Alpha 1", "Alpha 2", "Beta 1"],
            "train_name": ["Train 1", "Train 2", "Train 1"],
            "train": [1, 2, 1],
            "status": ["Under Construction", "Under Construction", "Cancelled"],
            "capacity_mtpa": [2.0, 3.0, 1.0],
            "plant_mapping_applied": [True, True, False],
            "train_mapping_applied": [True, True, False],
        }
    )
    full_change_df = capacity._build_ea_change_log(
        raw_df,
        ["Country A"],
        "exclude",
        None,
        None,
    )

    expected = capacity._build_ea_change_log(
        raw_df,
        ["Country A"],
        "exclude",
        "2026-03-01",
        "2026-03-01",
    )
    actual = capacity._build_ea_change_log(
        full_change_df,
        ["Country A"],
        "exclude",
        "2026-03-01",
        "2026-03-01",
    )

    pd.testing.assert_frame_equal(actual, expected.reset_index(drop=True))


def test_ea_retired_row_without_retirement_date_is_not_a_negative_event():
    raw_df = pd.DataFrame(
        {
            "month": pd.to_datetime(["1981-01-01", "2013-07-01"]),
            "country_name": ["Algeria", "Algeria"],
            "plant_name": ["Skikda GL2K", "Skikda GL1K"],
            "project_name": ["Skikda GL2K", "Skikda GL1K"],
            "train_name": ["Train 40", "Train 1 (rebuild)"],
            "train": [40, 1],
            "status": ["Retired", "Active"],
            "capacity_mtpa": [1.0, 4.5],
            "plant_mapping_applied": [False, False],
            "train_mapping_applied": [False, False],
        }
    )

    events = capacity._build_ea_change_log(
        raw_df, ["Algeria"], "exclude", None, None
    )

    assert events["train_name"].tolist() == ["Train 1 (rebuild)"]
    assert events["EA Net Delta (MTPA)"].tolist() == [4.5]
    assert events["EA Reductions (MTPA)"].tolist() == [0.0]


def test_source_loader_uses_completed_display_key(monkeypatch):
    expected = {"snapshot_key": "completed-key", "error_message": None}
    monkeypatch.setattr(
        capacity,
        "_fetch_capacity_source_state",
        lambda: {
            "source_key": "current-key",
            "display_source_key": "completed-key",
            "current_status": "running",
            "watermarks": {},
        },
    )
    monkeypatch.setattr(
        capacity,
        "_fetch_relational_capacity_source_snapshot",
        lambda source_key, _watermarks=None, source_state=None: (
            expected if source_key == "completed-key" else None
        ),
    )

    assert capacity._load_capacity_source_snapshot() is expected
    assert "preceding completed" in expected["error_message"]


def test_initial_load_uses_completed_cache_before_watermark_discovery(monkeypatch):
    monkeypatch.setattr(
        capacity,
        "fetch_capacity_source_state",
        lambda *_args: {
            "display_source_key": "completed-key",
            "display_finished_at": "2026-07-17T00:00:00Z",
        },
    )
    monkeypatch.setattr(
        capacity,
        "_fetch_relational_capacity_source_snapshot",
        lambda *_args, **_kwargs: {
            "snapshot_key": "completed-key",
            "available_countries": ["Qatar"],
            "metadata": {},
            "error_message": None,
            "min_date": "2026-01-01",
            "max_date": "2031-12-01",
            "last_success_at": "2026-07-17T00:00:00Z",
        },
    )
    monkeypatch.setattr(
        capacity,
        "get_available_capacity_scenarios",
        lambda _engine: pd.DataFrame(),
    )

    result = capacity.load_capacity_source_data(
        {"load": 1}, None, None, None
    )

    assert result[5] is None
    assert result[14]["discover_current"] is True
    assert result[15] is False


def test_source_freshness_failure_keeps_displayed_cache_and_stops_polling(monkeypatch):
    monkeypatch.setattr(
        capacity,
        "_fetch_capacity_source_state",
        lambda: (_ for _ in ()).throw(RuntimeError("watermark unavailable")),
    )
    source_ref = {
        "format": capacity.CAPACITY_SOURCE_REF_FORMAT,
        "source_key": "completed-key",
        "snapshot_key": "completed-key",
        "component": "train_store",
        "last_success_at": "2026-07-17T00:00:00Z",
    }

    result = capacity.poll_capacity_source_refresh(
        1,
        {"status": "running", "discover_current": True, "source_key": ""},
        None,
        None,
        None,
        source_ref,
    )

    assert "watermark unavailable" in result[5]
    assert result[14]["status"] == "failed"
    assert result[15] is True


def test_initial_source_failure_without_cache_stops_polling(monkeypatch):
    monkeypatch.setattr(
        capacity,
        "fetch_capacity_source_state",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("cache unavailable")),
    )
    monkeypatch.setattr(
        capacity,
        "get_available_capacity_scenarios",
        lambda _engine: pd.DataFrame(),
    )

    result = capacity.load_capacity_source_data(
        {"load": 1}, None, None, None
    )

    assert "completed Capacity source cache" in result[5]
    assert "cache unavailable" in result[5]
    assert result[14]["status"] == "failed"
    assert result[15] is True


def test_initial_scenario_failure_does_not_block_source_discovery(monkeypatch):
    monkeypatch.setattr(
        capacity,
        "fetch_capacity_source_state",
        lambda *_args: {},
    )
    monkeypatch.setattr(
        capacity,
        "get_available_capacity_scenarios",
        lambda _engine: (_ for _ in ()).throw(RuntimeError("scenario unavailable")),
    )

    result = capacity.load_capacity_source_data({"load": 1}, None, None, None)

    assert "scenario unavailable" in result[5]
    assert result[14]["discover_current"] is True
    assert result[15] is False


def test_missing_source_refresh_job_stops_polling_with_visible_error(monkeypatch):
    monkeypatch.setattr(capacity, "_read_capacity_refresh_job", lambda _key: None)

    result = capacity.poll_capacity_source_refresh(
        1,
        {"status": "running", "source_key": "missing-key"},
        None,
        None,
        None,
        None,
    )

    assert "job row is missing" in result[5]
    assert result[14]["status"] == "failed"
    assert result[15] is True


def test_orphaned_running_refresh_times_out_and_stops_polling(monkeypatch):
    failures = []
    monkeypatch.setattr(
        capacity,
        "_read_capacity_refresh_job",
        lambda _key: {
            "source_key": "source-key",
            "status": "running",
            "requested_at": "2000-01-01T00:00:00Z",
        },
    )
    monkeypatch.setattr(
        capacity,
        "mark_capacity_source_failed",
        lambda _engine, source_key, message: failures.append((source_key, message)),
    )

    result = capacity.poll_capacity_source_refresh(
        1,
        {"status": "running", "source_key": "source-key"},
        None,
        None,
        None,
        None,
    )

    assert failures == [
        ("source-key", "Refresh worker did not complete within five minutes.")
    ]
    assert result[14]["status"] == "failed"
    assert result[15] is True


def test_source_refresh_job_read_failure_stops_polling_with_visible_error(monkeypatch):
    monkeypatch.setattr(
        capacity,
        "_read_capacity_refresh_job",
        lambda _key: (_ for _ in ()).throw(RuntimeError("database unavailable")),
    )

    result = capacity.poll_capacity_source_refresh(
        1,
        {"status": "running", "source_key": "source-key"},
        None,
        None,
        None,
        None,
    )

    assert "database unavailable" in result[5]
    assert result[14]["status"] == "failed"
    assert result[15] is True


def test_failed_refresh_activates_already_completed_successor(monkeypatch):
    monkeypatch.setattr(
        capacity,
        "_read_capacity_refresh_job",
        lambda _key: {"source_key": "old-key", "status": "failed"},
    )
    monkeypatch.setattr(
        capacity,
        "_fetch_capacity_source_state",
        lambda: {"source_key": "new-key"},
    )
    monkeypatch.setattr(
        capacity,
        "_queue_capacity_source_refresh",
        lambda _key: {
            "source_key": "new-key",
            "status": "completed",
            "finished_at": "2026-07-17T00:00:00Z",
        },
    )
    monkeypatch.setattr(
        capacity,
        "_fetch_relational_capacity_source_snapshot",
        lambda *_args, **_kwargs: {
            "snapshot_key": "new-key",
            "available_countries": ["Qatar"],
            "metadata": {},
            "error_message": None,
            "min_date": "2026-01-01",
            "max_date": "2031-12-01",
            "last_success_at": "2026-07-17T00:00:00Z",
        },
    )
    monkeypatch.setattr(
        capacity,
        "get_available_capacity_scenarios",
        lambda _engine: pd.DataFrame(),
    )

    result = capacity.poll_capacity_source_refresh(
        1,
        {"status": "running", "source_key": "old-key"},
        None,
        None,
        None,
        None,
    )

    assert result[1]["source_key"] == "new-key"
    assert result[14]["status"] == "completed"
    assert result[15] is True


def test_failed_refresh_reports_successor_queue_failure(monkeypatch):
    monkeypatch.setattr(
        capacity,
        "_read_capacity_refresh_job",
        lambda _key: {"source_key": "old-key", "status": "failed"},
    )
    monkeypatch.setattr(
        capacity,
        "_fetch_capacity_source_state",
        lambda: {"source_key": "new-key"},
    )
    monkeypatch.setattr(
        capacity,
        "_queue_capacity_source_refresh",
        lambda _key: (_ for _ in ()).throw(RuntimeError("queue unavailable")),
    )

    result = capacity.poll_capacity_source_refresh(
        1,
        {"status": "running", "source_key": "old-key"},
        None,
        None,
        None,
        None,
    )

    assert "queue unavailable" in result[5]
    assert result[14]["status"] == "failed"
    assert result[15] is True


def test_source_reference_singleflight_reads_events_once(monkeypatch):
    capacity._CAPACITY_SOURCE_CACHE.clear()
    calls = []
    events = pd.DataFrame(
        [{
            "provider_name": "woodmac",
            "effective_month": "2026-01-01",
            "source_capacity_change_mtpa": 1.0,
            "allocation_share": 1.0,
            "country_name": "Country A",
        }]
    )

    def fake_fetch(_engine, source_key):
        calls.append(source_key)
        time.sleep(0.05)
        return events.copy()

    monkeypatch.setattr(capacity, "fetch_capacity_source_events", fake_fetch)
    monkeypatch.setattr(
        capacity,
        "_capacity_events_to_woodmac_change_log",
        lambda _events: pd.DataFrame(),
    )
    monkeypatch.setattr(
        capacity,
        "_capacity_events_to_ea_change_log",
        lambda _events: pd.DataFrame(),
    )
    ref = {
        "format": capacity.CAPACITY_SOURCE_REF_FORMAT,
        "source_key": "source-a",
        "snapshot_key": "source-a",
        "component": "train_store",
    }

    with ThreadPoolExecutor(max_workers=4) as executor:
        snapshots = list(
            executor.map(
                lambda _index: capacity._resolve_capacity_source_ref(ref, "train_store"),
                range(4),
            )
        )

    assert len(calls) == 1
    assert all(snapshot is snapshots[0] for snapshot in snapshots)


def test_initial_relational_snapshot_seeds_all_source_components(monkeypatch):
    capacity._CAPACITY_SOURCE_CACHE.clear()
    event_reads = []
    state_reads = []
    events = pd.DataFrame(
        [
            {
                "provider_name": "woodmac",
                "effective_month": "2026-01-01",
                "source_capacity_change_mtpa": 1.0,
                "allocation_share": 1.0,
                "country_name": "Country A",
            }
        ]
    )

    def fake_events(_engine, source_key):
        event_reads.append(source_key)
        return events.copy()

    def fake_state(_engine, source_key):
        state_reads.append(source_key)
        return {}

    monkeypatch.setattr(capacity, "fetch_capacity_source_events", fake_events)
    monkeypatch.setattr(capacity, "fetch_capacity_source_state", fake_state)
    monkeypatch.setattr(
        capacity,
        "_capacity_events_to_woodmac_change_log",
        lambda _events: pd.DataFrame(),
    )
    monkeypatch.setattr(
        capacity,
        "_capacity_events_to_ea_change_log",
        lambda _events: pd.DataFrame(),
    )
    source_state = {"display_finished_at": "2026-07-17T00:00:00Z"}

    snapshot = capacity._fetch_relational_capacity_source_snapshot(
        "source-a",
        {},
        source_state=source_state,
    )
    for component in ("woodmac_store", "train_store", "ea_store"):
        ref = capacity._build_capacity_source_ref(snapshot, component)
        capacity._resolve_capacity_source_ref(ref, component)

    assert event_reads == ["source-a"]
    assert state_reads == []


def test_initial_relational_snapshot_build_is_singleflight(monkeypatch):
    capacity._CAPACITY_SOURCE_CACHE.clear()
    calls = []
    events = pd.DataFrame(
        [{
            "provider_name": "woodmac",
            "effective_month": "2026-01-01",
            "source_capacity_change_mtpa": 1.0,
            "allocation_share": 1.0,
            "country_name": "Country A",
        }]
    )

    def fake_fetch(_engine, source_key):
        calls.append(source_key)
        time.sleep(0.05)
        return events.copy()

    monkeypatch.setattr(capacity, "fetch_capacity_source_events", fake_fetch)
    monkeypatch.setattr(
        capacity,
        "_capacity_events_to_woodmac_change_log",
        lambda _events: pd.DataFrame(),
    )
    monkeypatch.setattr(
        capacity,
        "_capacity_events_to_ea_change_log",
        lambda _events: pd.DataFrame(),
    )
    state = {"display_finished_at": "2026-07-17T00:00:00Z"}

    with ThreadPoolExecutor(max_workers=4) as executor:
        snapshots = list(
            executor.map(
                lambda _index: capacity._fetch_relational_capacity_source_snapshot(
                    "source-singleflight", {}, source_state=state
                ),
                range(4),
            )
        )

    assert calls == ["source-singleflight"]
    assert all(snapshot["event_count"] == 1 for snapshot in snapshots)


def test_capacity_action_state_requires_selection_dirty_and_clean_publish_status():
    clean_run = {
        "capacity_scenario_id": 2,
        "run_status": "completed",
        "monthly_row_count": 61,
        "blocking_qa_count": 0,
        "is_stale": False,
        "is_current_published": False,
    }

    no_scenario = capacity._build_capacity_scenario_action_state(
        None, {"dirty": False}, None, clean_run, None, None, None
    )
    assert all(no_scenario.values())

    dirty = capacity._build_capacity_scenario_action_state(
        2, {"dirty": True}, None, clean_run, "train", "profile", None
    )
    assert dirty["save_disabled"] is False
    assert dirty["revert_disabled"] is False
    assert dirty["generate_disabled"] is True
    assert dirty["publish_disabled"] is True
    assert dirty["profile_disabled"] is True

    clean = capacity._build_capacity_scenario_action_state(
        2, {"dirty": False}, None, clean_run, "train", "profile", None
    )
    assert clean["save_disabled"] is True
    assert clean["generate_disabled"] is False
    assert clean["publish_disabled"] is False
    assert clean["profile_disabled"] is False

    other_scenario = capacity._build_capacity_scenario_action_state(
        3, {"dirty": False}, None, clean_run, "train", "profile", None
    )
    assert other_scenario["publish_disabled"] is True

    clean_run["is_current_published"] = True
    published = capacity._build_capacity_scenario_action_state(
        2, {"dirty": False}, None, clean_run, "train", "profile", None
    )
    assert published["publish_disabled"] is True


def test_scenario_mapping_preflight_reports_exact_unresolved_rows():
    rows = pd.DataFrame(
        [
            {
                "train_key": None,
                "base_provider": "energy_aspects",
                "country_name": "Australia",
                "plant_name": "Ichthys LNG",
                "train_label": "Debottlenecking",
                "base_first_date": "2023-12-01",
            },
            {
                "train_key": "resolved",
                "base_provider": "energy_aspects",
                "country_name": "Mexico",
                "plant_name": "Altamira",
                "train_label": "Train 1",
                "base_first_date": "2024-07-01",
            },
        ]
    )

    try:
        capacity._validate_capacity_scenario_mapping_preflight(
            rows, "energy_aspects"
        )
    except ValueError as exc:
        message = str(exc)
        assert "1 provider event" in message
        assert "Ichthys LNG" in message
        assert "2023-12-01" in message
    else:
        raise AssertionError("Expected unresolved scenario creation to be blocked.")


def test_provider_scenario_preflight_cannot_be_bypassed_by_name_matching(monkeypatch):
    unresolved = pd.DataFrame(
        [{
            "train_key": None,
            "base_provider": "energy_aspects",
            "country_name": "Algeria",
            "plant_name": "Bethioua GL1Z",
            "train_label": "Train 1",
            "base_first_date": "2026-01-01",
        }]
    )
    monkeypatch.setattr(
        capacity,
        "attach_registry_keys_to_capacity_rows",
        lambda rows, _engine: rows.assign(train_key="name-matched-train"),
    )

    with pytest.raises(ValueError, match="without a canonical train mapping"):
        capacity._attach_and_validate_capacity_scenario_keys(
            unresolved,
            "energy_aspects",
        )


def test_source_key_change_selects_new_completed_cache(monkeypatch):
    states = iter(
        [
            {"source_key": "a", "display_source_key": "a", "watermarks": {}},
            {"source_key": "b", "display_source_key": "b", "watermarks": {}},
        ]
    )
    monkeypatch.setattr(capacity, "_fetch_capacity_source_state", lambda: next(states))
    monkeypatch.setattr(
        capacity,
        "_fetch_relational_capacity_source_snapshot",
        lambda source_key, _watermarks=None, source_state=None: {
            "snapshot_key": source_key,
            "error_message": None,
        },
    )

    assert capacity._load_capacity_source_snapshot()["snapshot_key"] == "a"
    assert capacity._load_capacity_source_snapshot()["snapshot_key"] == "b"


def test_source_reference_contains_no_revision_or_request_id():
    ref = capacity._build_capacity_source_ref(
        {"snapshot_key": "source-a"}, "train_store"
    )

    assert ref["source_key"] == "source-a"
    assert "revision" not in ref
    assert "request_id" not in ref


def test_source_loader_requires_a_completed_fallback(monkeypatch):
    monkeypatch.setattr(
        capacity,
        "_fetch_capacity_source_state",
        lambda: {"source_key": "source", "display_source_key": ""},
    )

    try:
        capacity._load_capacity_source_snapshot()
    except RuntimeError as exc:
        assert "No completed relational" in str(exc)
    else:
        raise AssertionError("Expected a missing-cache error.")


def test_prepared_provider_data_is_reused_for_identical_filters(monkeypatch):
    capacity._CAPACITY_DERIVED_CACHE.clear()
    raw_df = _woodmac_monthly_fixture()
    events_df = capacity._build_train_change_log(
        raw_df,
        None,
        "rest_of_world",
        None,
        None,
    )
    train_store = events_df
    original_builder = capacity._build_train_change_log
    calls = []

    def counted_builder(*args, **kwargs):
        calls.append(1)
        return original_builder(*args, **kwargs)

    monkeypatch.setattr(capacity, "_build_train_change_log", counted_builder)

    first = capacity._get_prepared_capacity_provider_data(
        train_store,
        None,
        ["Country A"],
        "exclude",
        "2026-01-01",
        "2026-02-01",
    )
    first_call_count = len(calls)
    second = capacity._get_prepared_capacity_provider_data(
        train_store,
        None,
        ["Country A"],
        "exclude",
        "2026-01-01",
        "2026-02-01",
    )
    scenario_aware = capacity._get_prepared_capacity_provider_data(
        train_store,
        None,
        ["Country A"],
        "exclude",
        "2026-01-01",
        "2026-02-01",
        scenario_rows_df=_scenario_rows_fixture(),
    )

    assert first is second
    assert first_call_count > 0
    assert len(calls) == first_call_count
    assert scenario_aware["woodmac_change_df"] is first["woodmac_change_df"]


def test_woodmac_train_query_avoids_redundant_coverage_anti_joins():
    query = capacity.WOODMAC_TRAIN_CAPACITY_QUERY
    carry_forward_sql = query.split("monthly_carry_forward AS (", 1)[1]

    assert "LEFT JOIN monthly_coverage_map" not in carry_forward_sql
    assert "capacity_source" in query
    assert "'monthly_carry_forward'::text" in query


def test_source_key_tracks_all_behavioral_mapping_hashes():
    query = capacity.CAPACITY_SOURCE_WATERMARK_QUERY

    assert "country_mapping_hash" in query
    assert "plant_mapping_hash" not in query
    assert "mapping_plant_name" not in query
    assert "provider_link_hash" in query
    assert "canonical_registry_hash" in query
    assert "mapper_version" in query
    assert "train_mapping_hash" not in query
    assert (
        capacity.CAPACITY_PROVIDER_MAPPER_VERSION
        == "capacity_provider_mapper_v4_provider_links_only"
    )
    assert "fundamentals_capacity_source_snapshots" not in query


def test_provider_link_mapping_uses_canonical_country_and_terminal():
    raw = pd.DataFrame(
        [
            {
                "country_name": "Mauritania-Senegal JDZ",
                "plant_name": "Tortue Phase 2 FLNG",
                "raw_plant_name": "Tortue Phase 2 FLNG",
                "raw_train_name": "Train 1",
                "provider_plant_id": "848",
                "provider_train_id": "636",
                "capacity_mtpa": 1.75,
                "plant_mapping_applied": False,
            }
        ]
    )
    links = pd.DataFrame(
        [
            {
                "provider_name": "woodmac",
                "provider_plant_id": "848",
                "provider_train_id": "636",
                "train_key": "10000000-0000-0000-0000-000000000001",
                "terminal_key": "20000000-0000-0000-0000-000000000001",
                "allocation_share": 1.0,
                "canonical_country_name": "Mauritania",
                "canonical_terminal_name": "Tortue FLNG",
                "canonical_train_label": "Train 1",
            }
        ]
    )

    mapped = capacity._apply_provider_link_mapping(
        raw,
        links,
        provider_name="woodmac",
        provider_plant_id_column="provider_plant_id",
        provider_train_id_column="provider_train_id",
        parent_source_column="raw_plant_name",
        source_column="raw_train_name",
    )

    assert mapped.loc[0, "country_name"] == "Mauritania"
    assert mapped.loc[0, "plant_name"] == "Tortue FLNG"
    assert bool(mapped.loc[0, "train_mapping_applied"])
    assert mapped.loc[0, "train_key"] == "10000000-0000-0000-0000-000000000001"


def test_refresh_status_includes_last_success_time():
    status = capacity._build_capacity_refresh_status(
        "running",
        snapshot_key="source:4",
        last_success_at="2026-07-14T09:10:00+00:00",
    )

    assert "Source source:4" in status.children
    assert "Last success 2026-07-14 09:10 UTC" in status.children


def test_capacity_source_metadata_status_formats_utc_and_selected_scenario():
    metadata = {
        "ea": {"upload_timestamp_utc": "2026-03-31T12:27:25+04:00"},
        "woodmac": {
            "monthly_upload_timestamp_utc": "2026-07-14T11:09:55+00:00",
            "annual_upload_timestamp_utc": "2026-07-13T10:00:00+00:00",
        },
    }
    options = [
        {"scenario_id": 2, "scenario_name": "Base Case"},
        {"scenario_id": 7, "scenario_name": "Alternative"},
    ]

    status = capacity.render_capacity_source_metadata_status(metadata, 2, options)

    assert status == (
        "EA uploaded: 2026-03-31 08:27 UTC"
        " · WoodMac uploaded: 2026-07-14 11:09 UTC"
        " · Internal scenario: Base Case"
    )


def test_capacity_source_metadata_status_handles_missing_and_deleted_values():
    status = capacity.render_capacity_source_metadata_status(
        {"ea": {}, "woodmac": {}},
        99,
        [{"scenario_id": 2, "scenario_name": "Base Case"}],
    )

    assert status == (
        "EA uploaded: Unavailable"
        " · WoodMac uploaded: Unavailable"
        " · Internal scenario: None selected"
    )


def _scenario_rows_fixture() -> pd.DataFrame:
    rows = []
    for index in range(12):
        rows.append(
            {
                "scenario_row_key": f"row-{index}",
                "terminal_key": f"terminal-{index}",
                "train_key": f"train-{index}",
                "country_name": "Country A",
                "plant_name": "Alpha",
                "train_label": str(index + 1),
                "base_provider": "woodmac",
                "base_first_date": pd.Timestamp("2026-01-01") + pd.offsets.MonthBegin(index),
                "base_capacity_mtpa": float(index + 1),
                "scenario_first_date": pd.Timestamp("2026-01-01") + pd.offsets.MonthBegin(index),
                "scenario_capacity_mtpa": float(index + 1),
                "scenario_note": "",
                "display_sort_plant": "Alpha",
                "display_sort_train": index + 1,
            }
        )
    return capacity._prepare_capacity_scenario_rows_df(pd.DataFrame(rows))


def test_source_reference_resolves_exact_source_key(monkeypatch):
    capacity._CAPACITY_SOURCE_CACHE.clear()
    expected = pd.DataFrame({"Effective Date": ["2026-01-01"]})
    events = pd.DataFrame(
        [{
            "provider_name": "woodmac",
            "effective_month": "2026-01-01",
            "source_capacity_change_mtpa": 1.0,
            "allocation_share": 1.0,
            "country_name": "Country A",
        }]
    )
    fetch_calls = []

    def fake_fetch(_engine, source_key):
        fetch_calls.append(source_key)
        return events.copy()

    monkeypatch.setattr(capacity, "fetch_capacity_source_events", fake_fetch)
    monkeypatch.setattr(
        capacity, "_capacity_events_to_woodmac_change_log", lambda _events: expected
    )
    monkeypatch.setattr(
        capacity, "_capacity_events_to_ea_change_log", lambda _events: pd.DataFrame()
    )
    ref = capacity._build_capacity_source_ref(
        {"snapshot_key": "source-ref"}, "train_store"
    )
    restored = capacity._deserialize_train_capacity_store(ref)

    pd.testing.assert_frame_equal(restored, expected)
    assert fetch_calls == ["source-ref"]
    assert capacity._get_train_capacity_snapshot_key(ref) == "source-ref"


def test_scenario_working_v2_clean_and_delta_round_trip(monkeypatch):
    capacity._CAPACITY_SCENARIO_CACHE.clear()
    baseline_df = _scenario_rows_fixture()
    fetch_calls = []

    def fake_fetch(scenario_id, _engine, snapshot_timestamp_utc=None):
        fetch_calls.append(scenario_id)
        assert snapshot_timestamp_utc == "2026-07-14T00:00:00"
        return baseline_df.copy()

    monkeypatch.setattr(capacity, "fetch_capacity_scenario_rows", fake_fetch)
    options = [
        {
            "scenario_id": 7,
            "current_snapshot_timestamp_utc": "2026-07-14T00:00:00",
        }
    ]
    clean_store = capacity._build_capacity_scenario_working_store(7, options)
    clean_df = capacity._resolve_active_capacity_scenario_rows(clean_store)
    pd.testing.assert_frame_equal(clean_df, baseline_df)

    edited_df = baseline_df.copy()
    edited_df.loc[:9, "scenario_capacity_mtpa"] += 0.25
    edited_df = edited_df.iloc[:-1].copy()
    delta_store = capacity._build_capacity_scenario_delta_store(
        7,
        edited_df,
        options,
    )
    restored_df = capacity._resolve_active_capacity_scenario_rows(delta_store)

    pd.testing.assert_frame_equal(restored_df, edited_df.reset_index(drop=True))
    assert delta_store["format"] == capacity.CAPACITY_SCENARIO_WORKING_FORMAT
    assert delta_store["mode"] == "delta"
    assert delta_store["deleted_keys"] == ["row-11"]
    assert len(capacity._deserialize_dataframe(delta_store["upserts"])) == 10
    assert len(json.dumps(delta_store)) < 150_000
    assert fetch_calls == [7]


def test_capacity_callbacks_no_longer_upload_timeline_rowdata_as_state():
    source_text = Path(capacity.__file__).read_text()
    assert 'State("capacity-page-train-timeline-table", "rowData")' not in source_text
    assert 'Input("capacity-page-train-timeline-table", "cellValueChanged")' in source_text


def test_refresh_worker_completes_one_source_key(monkeypatch):
    class FakeConnection:
        def __enter__(self): return self
        def __exit__(self, *_args): return False
        def execute(self, *_args, **_kwargs): return None
        def commit(self): return None

    class FakeEngine:
        def connect(self): return FakeConnection()

    completed = []
    monkeypatch.setattr(capacity, "engine", FakeEngine())
    monkeypatch.setattr(
        capacity, "read_capacity_refresh_job", lambda *_args: {"status": "running"}
    )
    monkeypatch.setattr(
        capacity, "_fetch_capacity_source_state", lambda: {"source_key": "source-job"}
    )
    monkeypatch.setattr(
        capacity,
        "_build_relational_capacity_source_events",
        lambda _source_key: pd.DataFrame([{"source_key": "source-job"}]),
    )
    monkeypatch.setattr(
        capacity,
        "complete_capacity_source_refresh",
        lambda _engine, source_key, events: completed.append((source_key, len(events))),
    )

    capacity._run_capacity_source_refresh_job("source-job")

    assert completed == [("source-job", 1)]


def test_refresh_worker_failure_records_no_events(monkeypatch):
    class FakeConnection:
        def __enter__(self): return self
        def __exit__(self, *_args): return False
        def execute(self, *_args, **_kwargs): return None
        def commit(self): return None

    class FakeEngine:
        def connect(self): return FakeConnection()

    failures = []
    monkeypatch.setattr(capacity, "engine", FakeEngine())
    monkeypatch.setattr(
        capacity, "read_capacity_refresh_job", lambda *_args: {"status": "running"}
    )
    monkeypatch.setattr(
        capacity, "_fetch_capacity_source_state", lambda: {"source_key": "source-job"}
    )
    monkeypatch.setattr(
        capacity,
        "_build_relational_capacity_source_events",
        lambda _source_key: (_ for _ in ()).throw(RuntimeError("build failed")),
    )
    monkeypatch.setattr(
        capacity,
        "mark_capacity_source_failed",
        lambda _engine, source_key, message: failures.append((source_key, message)),
    )

    capacity._run_capacity_source_refresh_job("source-job")

    assert failures == [("source-job", "build failed")]


def test_refresh_queue_uses_source_key_without_request_id(monkeypatch):
    submitted = []

    class FakeExecutor:
        def submit(self, function, *args):
            submitted.append((function, args))

    monkeypatch.setattr(capacity, "_CAPACITY_REFRESH_EXECUTOR", FakeExecutor())
    calls = iter(
        [
            ({"source_key": "source-job", "status": "running"}, True),
            ({"source_key": "source-job", "status": "running"}, False),
        ]
    )
    monkeypatch.setattr(capacity, "read_capacity_refresh_job", lambda *_args: None)
    monkeypatch.setattr(capacity, "mark_capacity_source_running", lambda *_args: next(calls))

    first = capacity._queue_capacity_source_refresh("source-job")
    second = capacity._queue_capacity_source_refresh("source-job")

    assert first["source_key"] == second["source_key"] == "source-job"
    assert "request_id" not in first
    assert len(submitted) == 1
    assert submitted[0][1] == ("source-job",)


def test_refresh_queue_submission_failure_marks_job_failed(monkeypatch):
    failures = []

    class FailingExecutor:
        def submit(self, *_args):
            raise RuntimeError("executor unavailable")

    monkeypatch.setattr(capacity, "_CAPACITY_REFRESH_EXECUTOR", FailingExecutor())
    monkeypatch.setattr(capacity, "read_capacity_refresh_job", lambda *_args: None)
    monkeypatch.setattr(
        capacity,
        "mark_capacity_source_running",
        lambda *_args: (
            {"source_key": "source-job", "status": "running"},
            True,
        ),
    )
    monkeypatch.setattr(
        capacity,
        "mark_capacity_source_failed",
        lambda _engine, source_key, message: failures.append((source_key, message)),
    )

    with pytest.raises(RuntimeError, match="executor unavailable"):
        capacity._queue_capacity_source_refresh("source-job")

    assert failures == [
        (
            "source-job",
            "Capacity source refresh worker could not start: executor unavailable",
        )
    ]


def test_unavailable_persisted_scenario_falls_back_to_base_case():
    options = [
        {"scenario_id": 2, "scenario_name": "Base Case"},
        {"scenario_id": 7, "scenario_name": "Alternative"},
    ]

    assert capacity.sync_capacity_scenario_dropdown_value(options, None, 99) == 2
    assert capacity.sync_capacity_scenario_dropdown_value(options, 99, 99) == 2
    assert capacity.sync_capacity_scenario_dropdown_value(options, None, 7) is capacity.no_update


def test_options_load_initializes_selected_and_working_scenario_stores(monkeypatch):
    options = [
        {
            "scenario_id": 2,
            "scenario_name": "Base Case",
            "current_snapshot_timestamp_utc": "2026-07-17T00:00:00Z",
        }
    ]
    rows = pd.DataFrame(
        [
            {
                "scenario_row_key": "event-1",
                "train_key": "train-1",
                "country_name": "Qatar",
                "plant_name": "Ras Laffan",
                "train_label": "Train 1",
            }
        ]
    )
    monkeypatch.setattr(
        capacity,
        "ctx",
        SimpleNamespace(
            triggered_id="capacity-page-capacity-scenario-options-store"
        ),
    )
    monkeypatch.setattr(
        capacity,
        "fetch_capacity_scenario_rows",
        lambda *_args, **_kwargs: rows.copy(),
    )
    monkeypatch.setattr(
        capacity,
        "_cache_saved_capacity_scenario_rows",
        lambda *_args, **_kwargs: None,
    )

    result = capacity.handle_capacity_scenario_selection(
        None,
        options,
        None,
        {"dirty": False},
    )

    assert result[0] == 2
    assert result[1]["scenario_id"] == 2
    assert result[1]["base_revision"] == "2026-07-17T00:00:00Z"
    assert result[2] == {"dirty": False}


def test_options_refresh_preserves_valid_dirty_scenario(monkeypatch):
    options = [{"scenario_id": 2, "scenario_name": "Base Case"}]
    monkeypatch.setattr(
        capacity,
        "ctx",
        SimpleNamespace(
            triggered_id="capacity-page-capacity-scenario-options-store"
        ),
    )

    with pytest.raises(PreventUpdate):
        capacity.handle_capacity_scenario_selection(
            2,
            options,
            2,
            {"dirty": True},
        )


def test_failed_scenario_switch_restores_visible_dropdown_target(monkeypatch):
    options = [
        {"scenario_id": 2, "scenario_name": "Base Case"},
        {"scenario_id": 3, "scenario_name": "Alternative"},
    ]
    monkeypatch.setattr(
        capacity,
        "ctx",
        SimpleNamespace(triggered_id="capacity-page-internal-scenario-dropdown"),
    )
    monkeypatch.setattr(
        capacity,
        "fetch_capacity_scenario_rows",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("scenario unavailable")
        ),
    )

    result = capacity.handle_capacity_scenario_selection(
        3,
        options,
        2,
        {"dirty": False},
    )

    assert result[0] is capacity.no_update
    assert result[6] == 2


def test_failed_first_scenario_load_clears_visible_dropdown(monkeypatch):
    options = [{"scenario_id": 2, "scenario_name": "Base Case"}]
    monkeypatch.setattr(
        capacity,
        "ctx",
        SimpleNamespace(triggered_id="capacity-page-internal-scenario-dropdown"),
    )
    monkeypatch.setattr(
        capacity,
        "fetch_capacity_scenario_rows",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("scenario unavailable")
        ),
    )

    result = capacity.handle_capacity_scenario_selection(
        2,
        options,
        None,
        {"dirty": False},
    )

    assert result[6] == {"clear": True}
    assert capacity.sync_capacity_scenario_dropdown_value(
        options, result[6], 2
    ) is None


def test_source_refresh_reloads_current_scenario_pointer(monkeypatch):
    rows = pd.DataFrame(
        [
            {
                "scenario_row_key": "event-1",
                "upload_timestamp_utc": "2026-07-17T02:00:00Z",
            }
        ]
    )
    calls = []

    def fake_fetch(scenario_id, _engine, **kwargs):
        calls.append((scenario_id, kwargs))
        return rows.copy()

    monkeypatch.setattr(capacity, "fetch_capacity_scenario_rows", fake_fetch)
    monkeypatch.setattr(
        capacity,
        "_cache_saved_capacity_scenario_rows",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        capacity,
        "_build_capacity_scenario_working_store",
        lambda scenario_id, _options, base_revision=None: {
            "scenario_id": scenario_id,
            "base_revision": base_revision,
        },
    )

    result = capacity.refresh_selected_capacity_scenario_working_copy(
        "refresh",
        2,
        {"dirty": False},
        [
            {
                "scenario_id": 2,
                "current_snapshot_timestamp_utc": "2026-07-17T01:00:00Z",
            }
        ],
        None,
        None,
    )

    assert calls == [(2, {})]
    assert result[0]["base_revision"] == "2026-07-17T02:00:00+00:00"


def test_source_refresh_scenario_read_failure_is_visible(monkeypatch):
    monkeypatch.setattr(
        capacity,
        "fetch_capacity_scenario_rows",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("scenario database unavailable")
        ),
    )

    result = capacity.refresh_selected_capacity_scenario_working_copy(
        "refresh",
        2,
        {"dirty": False},
        [],
        None,
        None,
    )

    assert result[0] is capacity.no_update
    assert "scenario database unavailable" in str(result[2])


def test_confirmed_scenario_switch_failure_restores_current_selection(monkeypatch):
    monkeypatch.setattr(
        capacity,
        "ctx",
        SimpleNamespace(
            triggered=[
                {
                    "prop_id": (
                        "capacity-page-capacity-scenario-switch-confirm."
                        "submit_n_clicks"
                    )
                }
            ]
        ),
    )
    monkeypatch.setattr(
        capacity,
        "_resolve_active_capacity_scenario_rows",
        lambda *_args, **_kwargs: pd.DataFrame(),
    )
    monkeypatch.setattr(
        capacity,
        "_get_capacity_scenario_base_revision",
        lambda *_args, **_kwargs: "2026-07-17T00:00:00Z",
    )
    monkeypatch.setattr(
        capacity,
        "fetch_capacity_scenario_rows",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("scenario database unavailable")
        ),
    )

    result = capacity.manage_capacity_scenario_state(
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        2,
        {},
        {"dirty": True},
        3,
        [],
        None,
        None,
        None,
        None,
        None,
        None,
    )

    assert result[0] is capacity.no_update
    assert result[4] is None
    assert result[5] == 2
    assert "scenario database unavailable" in str(result[6])


def test_scenario_option_recovery_clears_only_its_error(monkeypatch):
    monkeypatch.setattr(
        capacity,
        "get_available_capacity_scenarios",
        lambda _engine: pd.DataFrame(),
    )

    recovered = capacity.load_capacity_scenario_options(
        "refresh",
        "Internal scenario options could not be refreshed: temporary failure",
    )
    unrelated = capacity.load_capacity_scenario_options(
        "refresh",
        "Capacity source cache is unavailable",
    )

    assert recovered[1] is None
    assert unrelated[1] is capacity.no_update


def test_command_running_store_disables_all_scenario_actions():
    result = capacity.sync_capacity_scenario_controls(
        [{"scenario_id": 2, "scenario_name": "Base Case"}],
        2,
        {"dirty": True},
        None,
        {
            "capacity_scenario_id": 2,
            "run_status": "completed",
            "monthly_row_count": 1,
        },
        "train-key",
        "profile-id",
        None,
        True,
    )

    assert all(
        result[index] is True
        for index in [4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16]
    )


def test_filter_components_use_session_persistence():
    component_ids = {
        "capacity-page-country-dropdown",
        "capacity-page-other-country-mode",
        "capacity-page-top-table-view",
        "capacity-page-date-range",
        "capacity-page-train-change-time-view",
        "capacity-page-train-change-view-mode",
        "capacity-page-train-timeline-original-name-visibility",
        "capacity-page-train-timeline-chart-source",
        "capacity-page-train-timeline-chart-compare",
        "capacity-page-internal-scenario-dropdown",
    }
    found = {}

    def walk(component):
        if hasattr(component, "to_plotly_json"):
            props = component.to_plotly_json().get("props", {})
            component_id = props.get("id")
            if component_id in component_ids:
                found[component_id] = props
            walk(props.get("children"))
        elif isinstance(component, (list, tuple)):
            for child in component:
                walk(child)

    walk(capacity.layout)
    assert set(found) == component_ids
    assert all(props.get("persistence") is True for props in found.values())
    assert all(props.get("persistence_type") == "session" for props in found.values())


def test_selected_capacity_scenario_is_the_timeline_chart_source():
    scenario_options = [
        {"scenario_id": 2, "scenario_name": "Base Case"},
        {
            "scenario_id": 11,
            "scenario_name": "WoodMac with Screenshot Overrides",
        },
    ]

    source_options, source_value, compare_options, compare_value = (
        capacity.sync_train_timeline_chart_controls(
            11,
            scenario_options,
            "woodmac",
            "energy_aspects",
        )
    )

    assert source_value == "internal_scenario"
    assert compare_value == "woodmac"
    assert source_options == compare_options
    assert source_options[-1] == {
        "label": "WoodMac with Screenshot Overrides",
        "value": "internal_scenario",
    }


def test_timeline_chart_reuses_cached_figure_for_identical_rows(monkeypatch):
    capacity._CAPACITY_VIEW_CACHE.clear()
    capacity._CAPACITY_VIEW_CACHE_KEY_LOCKS.clear()
    calls = []

    def build_figure(*_args, **_kwargs):
        calls.append(1)
        return {"figure": len(calls)}

    monkeypatch.setattr(
        capacity,
        "_create_train_timeline_comparison_figure",
        build_figure,
    )
    rows = [{"Country": "Qatar", "Plant": "Ras Laffan", "Train": "1"}]
    options = [
        {"label": "Woodmac", "value": "woodmac"},
        {"label": "Energy Aspects", "value": "energy_aspects"},
    ]

    first = capacity.render_train_timeline_comparison_chart(
        "woodmac", "energy_aspects", "2026-01-01", "2031-12-01", rows, options
    )
    second = capacity.render_train_timeline_comparison_chart(
        "woodmac", "energy_aspects", "2026-01-01", "2031-12-01", rows, options
    )
    note_only = capacity.render_train_timeline_comparison_chart(
        "woodmac",
        "energy_aspects",
        "2026-01-01",
        "2031-12-01",
        [{**rows[0], "Scenario Note": "Review text only"}],
        options,
    )
    changed = capacity.render_train_timeline_comparison_chart(
        "woodmac",
        "energy_aspects",
        "2026-01-01",
        "2031-12-01",
        rows + [{"Country": "Oman", "Plant": "Oman LNG", "Train": "1"}],
        options,
    )

    assert first is second
    assert note_only is first
    assert changed is not first
    assert len(calls) == 2


def test_timeline_grid_avoids_dynamic_content_autosizing():
    wrapper = capacity._create_train_timeline_table(
        "timeline-grid-test",
        pd.DataFrame(),
    )
    grid = next(
        child
        for child in wrapper.children
        if getattr(child, "id", None) == "timeline-grid-test"
    )
    props = grid.to_plotly_json()["props"]

    assert "columnSize" not in props
    assert "columnSizeOptions" not in props
    assert "autoSizeStrategy" not in props["dashGridOptions"]


def test_ramp_status_card_shows_horizon_and_limits_qa_messages():
    card = capacity._render_capacity_ramp_status_card(
        {
            "run_id": 41,
            "capacity_scenario_id": 11,
            "run_status": "completed",
            "generator_version": "terminal_ramp_sql_inputs_v4_3",
            "generated_at": "2026-07-17T01:00:00Z",
            "horizon_start_month": "2026-07-01",
            "horizon_end_month": "2031-07-01",
            "blocking_qa_count": 0,
            "warning_qa_count": 4,
            "monthly_row_count": 27084,
            "qa_issues": [
                {"message": f"warning {index}"} for index in range(1, 5)
            ],
        }
    )
    rendered = str(card)

    assert "2026-07-01 to 2031-07-01" in rendered
    assert "warning 1" in rendered
    assert "warning 3" in rendered
    assert "warning 4" not in rendered


def test_train_timeline_uses_train_key_when_display_labels_differ():
    train_key = "0deb2115-130c-4a4d-87fd-dd40bcb21aef"
    timeline_df = pd.DataFrame(
        [
            {
                "scenario_row_key": "provider-row",
                "timeline_reference_key": "provider-timeline-row",
                "Country": "Australia",
                "Plant": "Pluto",
                "Train": "2",
                "train_key": train_key,
                "Woodmac Original Name": "Pluto Expansion Train 2",
                "Woodmac FID Date": "2021-11-21",
                "Woodmac First Date": "2026-11-01",
                "Woodmac Capacity Change": 5.0,
                "Energy Aspects Original Plant": "Pluto",
                "Energy Aspects Original Train": "Train 2",
                "Energy Aspects First Date": "2026-10-01",
                "Energy Aspects Capacity Change": 4.9,
                "timeline_direction": "addition",
                "display_sort_plant": "Pluto",
                "display_sort_train": 2,
                "display_sort_effective_date": pd.Timestamp("2026-10-01"),
                "display_sort_direction": 1,
            }
        ]
    )
    scenario_rows_df = pd.DataFrame(
        [
            {
                "scenario_row_key": "scenario-row",
                "train_key": train_key,
                "country_name": "Australia",
                "plant_name": "Pluto",
                "train_label": "Train 2",
                "base_provider": "woodmac",
                "base_first_date": pd.Timestamp("2026-11-01"),
                "base_capacity_mtpa": 5.0,
                "scenario_first_date": pd.Timestamp("2026-10-01"),
                "scenario_capacity_mtpa": 5.0,
                "scenario_note": "Screenshot override",
                "display_sort_plant": "Pluto",
                "display_sort_train": 2,
            }
        ]
    )

    grid_df = capacity._build_train_timeline_grid_rows(
        timeline_df,
        scenario_rows_df=scenario_rows_df,
        aggregate_from_date="2025-12-01",
    )

    assert len(grid_df) == 1
    row = grid_df.iloc[0]
    assert row["Train"] == "Train 2"
    assert row["train_key"] == train_key
    assert row["Woodmac First Date"] == "2026-11-01"
    assert row["Scenario First Date"] == "2026-10-01"

    figure = capacity._create_train_timeline_comparison_figure(
        grid_df.to_dict("records"),
        "internal_scenario",
        "woodmac",
        "2025-12-01",
        "2031-12-01",
        "Screenshot Scenario",
        "Woodmac",
    )
    assert sum(bool(annotation.showarrow) for annotation in figure.layout.annotations) == 1


def test_train_timeline_unresolved_fallback_normalizes_simple_train_labels():
    timeline_df = pd.DataFrame(
        [
            {
                "scenario_row_key": "provider-row",
                "timeline_reference_key": "provider-timeline-row",
                "Country": "Country A",
                "Plant": "Terminal A",
                "Train": "2",
                "train_key": None,
                "Woodmac First Date": "2028-01-01",
                "Woodmac Capacity Change": 1.0,
                "timeline_direction": "addition",
                "display_sort_plant": "Terminal A",
                "display_sort_train": 2,
                "display_sort_effective_date": pd.Timestamp("2028-01-01"),
                "display_sort_direction": 1,
            }
        ]
    )
    scenario_rows_df = pd.DataFrame(
        [
            {
                "scenario_row_key": "scenario-row",
                "train_key": None,
                "country_name": "Country A",
                "plant_name": "Terminal A",
                "train_label": "Train 2",
                "base_provider": "woodmac",
                "base_first_date": pd.Timestamp("2028-01-01"),
                "base_capacity_mtpa": 1.0,
                "scenario_first_date": pd.Timestamp("2027-12-01"),
                "scenario_capacity_mtpa": 1.0,
                "scenario_note": "Fallback test",
                "display_sort_plant": "Terminal A",
                "display_sort_train": 2,
            }
        ]
    )

    grid_df = capacity._build_train_timeline_grid_rows(
        timeline_df,
        scenario_rows_df=scenario_rows_df,
        aggregate_from_date="2025-12-01",
    )

    assert len(grid_df) == 1
    assert grid_df.iloc[0]["Woodmac First Date"] == "2028-01-01"
    assert grid_df.iloc[0]["Scenario First Date"] == "2027-12-01"


def test_train_timeline_preserves_event_level_frozen_provider_baselines():
    train_key = "297bd456-3b98-477e-b37b-09c8063c63fd"
    timeline_df = pd.DataFrame(
        [
            {
                "scenario_row_key": "provider-aggregate",
                "timeline_reference_key": "provider-timeline-row",
                "Country": "Congo",
                "Plant": "Congo FLNG Phase 2",
                "Train": "1",
                "train_key": train_key,
                "Woodmac First Date": "2026-02-01",
                "Woodmac Capacity Change": 2.4,
                "timeline_direction": "addition",
                "display_sort_plant": "Congo FLNG Phase 2",
                "display_sort_train": 1,
                "display_sort_effective_date": pd.Timestamp("2026-02-01"),
                "display_sort_direction": 1,
            }
        ]
    )
    scenario_rows_df = pd.DataFrame(
        [
            {
                "scenario_row_key": "scenario-february",
                "train_key": train_key,
                "country_name": "Congo",
                "plant_name": "Congo FLNG Phase 2",
                "train_label": "Train 1",
                "base_provider": "woodmac",
                "base_first_date": pd.Timestamp("2026-02-01"),
                "base_capacity_mtpa": 1.8,
                "scenario_first_date": pd.Timestamp("2026-02-01"),
                "scenario_capacity_mtpa": 1.8,
                "display_sort_plant": "Congo FLNG Phase 2",
                "display_sort_train": 1,
            },
            {
                "scenario_row_key": "scenario-march",
                "train_key": train_key,
                "country_name": "Congo",
                "plant_name": "Congo FLNG Phase 2",
                "train_label": "Train 1",
                "base_provider": "woodmac",
                "base_first_date": pd.Timestamp("2026-03-01"),
                "base_capacity_mtpa": 0.6,
                "scenario_first_date": pd.Timestamp("2026-03-01"),
                "scenario_capacity_mtpa": 0.6,
                "display_sort_plant": "Congo FLNG Phase 2",
                "display_sort_train": 1,
            },
        ]
    )

    grid_df = capacity._build_train_timeline_grid_rows(
        timeline_df,
        scenario_rows_df=scenario_rows_df,
        aggregate_from_date="2025-12-01",
    ).sort_values("Scenario First Date")

    assert list(grid_df["Woodmac First Date"]) == ["2026-02-01", "2026-03-01"]
    assert list(grid_df["Woodmac Capacity Change"]) == [1.8, 0.6]
    figure = capacity._create_train_timeline_comparison_figure(
        grid_df.to_dict("records"),
        "internal_scenario",
        "woodmac",
        "2025-12-01",
        "2031-12-01",
        "Scenario",
        "Woodmac",
    )
    assert sum(bool(annotation.showarrow) for annotation in figure.layout.annotations) == 0


def test_train_timeline_scenario_backfill_preserves_saved_blank_row_keys():
    train_key = "df39bd79-1207-5228-a2c5-66796d3d3589"
    timeline_df = pd.DataFrame(
        [
            {
                "scenario_row_key": "provider-row",
                "timeline_reference_key": "provider-timeline-row",
                "Country": "Congo",
                "Plant": "Tango FLNG",
                "Train": "Train 1",
                "train_key": train_key,
                "Energy Aspects First Date": "2026-01-01",
                "Energy Aspects Capacity Change": 2.4,
                "timeline_direction": "addition",
                "display_sort_plant": "Tango FLNG",
                "display_sort_train": 1,
                "display_sort_effective_date": pd.Timestamp("2026-01-01"),
                "display_sort_direction": 1,
            }
        ]
    )
    scenario_rows_df = pd.DataFrame(
        [
            {
                "scenario_row_key": row_key,
                "train_key": train_key,
                "country_name": "Congo",
                "plant_name": "Tango FLNG",
                "train_label": train_label,
                "base_provider": "energy_aspects",
                "base_first_date": pd.Timestamp("2026-01-01"),
                "base_capacity_mtpa": 2.4,
                "scenario_first_date": None,
                "scenario_capacity_mtpa": None,
                "scenario_note": "",
                "display_sort_plant": "Tango FLNG",
                "display_sort_train": 1,
            }
            for row_key, train_label in [
                ("saved-row-a", "1"),
                ("saved-row-b", "Train 1"),
            ]
        ]
    )
    scenario_lookup_df = pd.DataFrame(
        [
            {
                "scenario_row_key": "saved-row-a",
                "Country": "Congo",
                "Plant": "Tango FLNG",
                "Train": "Train 1",
                "train_key": train_key,
                "timeline_direction": "addition",
                "Scenario First Date": None,
                "Scenario Capacity": None,
                "Scenario Note": "",
                "__scenario_overridden": True,
                "lookup_is_out_of_range": False,
            }
        ]
    )

    grid_df = capacity._build_train_timeline_grid_rows(
        timeline_df,
        scenario_rows_df=scenario_rows_df,
        aggregate_from_date="2025-12-01",
        scenario_lookup_df=scenario_lookup_df,
    )

    assert set(grid_df["scenario_row_key"]) == {"saved-row-a", "saved-row-b"}
    assert grid_df["scenario_row_key"].is_unique


def test_train_timeline_scenario_backfill_updates_provider_only_row_identity():
    train_key = "df39bd79-1207-5228-a2c5-66796d3d3589"
    timeline_df = pd.DataFrame(
        [
            {
                "scenario_row_key": "provider-row",
                "timeline_reference_key": "provider-timeline-row",
                "Country": "Congo",
                "Plant": "Tango FLNG",
                "Train": "Train 1",
                "train_key": train_key,
                "Energy Aspects First Date": "2026-01-01",
                "Energy Aspects Capacity Change": 2.4,
                "timeline_direction": "addition",
                "display_sort_plant": "Tango FLNG",
                "display_sort_train": 1,
                "display_sort_effective_date": pd.Timestamp("2026-01-01"),
                "display_sort_direction": 1,
            }
        ]
    )
    scenario_lookup_df = pd.DataFrame(
        [
            {
                "scenario_row_key": "saved-scenario-row",
                "Country": "Congo",
                "Plant": "Tango FLNG",
                "Train": "Train 1",
                "train_key": train_key,
                "timeline_direction": "addition",
                "Scenario First Date": "2026-03-01",
                "Scenario Capacity": 2.4,
                "Scenario Note": "Out-of-range override",
                "__scenario_overridden": True,
                "lookup_is_out_of_range": True,
            }
        ]
    )

    grid_df = capacity._build_train_timeline_grid_rows(
        timeline_df,
        aggregate_from_date="2025-12-01",
        scenario_lookup_df=scenario_lookup_df,
    )

    row = grid_df.iloc[0]
    assert row["scenario_row_key"] == "saved-scenario-row"
    assert row["Scenario First Date"] == "2026-03-01"
    assert row["Scenario Capacity"] == 2.4
    assert bool(row["__scenario_out_of_range"])


def test_train_timeline_rejects_duplicate_grid_row_ids():
    duplicate_df = pd.DataFrame(
        {
            "scenario_row_key": ["duplicate-row", "duplicate-row"],
        }
    )

    with pytest.raises(ValueError, match="duplicate row id"):
        capacity._validate_train_timeline_grid_row_ids(duplicate_df)


# Consolidated from test_capacity_render_revisions.py.

import pandas as pd
import pytest
from dash._callback import GLOBAL_CALLBACK_MAP
from dash.exceptions import PreventUpdate

import index_shipping_snd
from pages import capacity


SOURCE_KEY = "capacity-source-2026-07-26"
BASE_REVISION = "2026-07-25T00:00:00"


def _source_ref(component, source_key=SOURCE_KEY):
    return {
        "format": capacity.CAPACITY_SOURCE_REF_FORMAT,
        "source_key": source_key,
        "snapshot_key": source_key,
        "component": component,
    }


def _source_revision(source_key=SOURCE_KEY):
    return {
        "format": "capacity_source_render_revision_v1",
        "source_key": source_key,
    }


def _scenario_state():
    selected = 7
    options = [
        {
            "scenario_id": selected,
            "scenario_name": "Base Case",
            "current_snapshot_timestamp_utc": BASE_REVISION,
        }
    ]
    working = {
        "format": capacity.CAPACITY_SCENARIO_WORKING_FORMAT,
        "scenario_id": selected,
        "base_revision": BASE_REVISION,
        "mode": "clean",
        "upserts": None,
        "deleted_keys": [],
        "row_order": [],
        "full_override": None,
    }
    revision = {
        "format": "capacity_scenario_render_revision_v1",
        "source_key": SOURCE_KEY,
        "scenario_id": str(selected),
        "base_revision": BASE_REVISION,
        "working_mode": "clean",
        "dirty": False,
        "dirty_source": "",
        "dirty_updated_at": "",
        "refresh_revision": "",
        "option_name": "Base Case",
        "option_revision": BASE_REVISION,
    }
    return revision, selected, working, options


def _walk_capacity_components(component):
    if component is None:
        return
    if isinstance(component, (list, tuple)):
        for child in component:
            yield from _walk_capacity_components(child)
        return
    yield component
    yield from _walk_capacity_components(getattr(component, "children", None))


def test_capacity_source_revision_requires_all_four_exact_components():
    woodmac = _source_ref("woodmac_store")
    train = _source_ref("train_store")
    ea = _source_ref("ea_store")
    metadata = {"snapshot_key": SOURCE_KEY}

    assert capacity._capacity_source_revision_is_coherent(
        _source_revision(),
        woodmac,
        train,
        ea,
        metadata,
    )
    assert not capacity._capacity_source_revision_is_coherent(
        _source_revision(),
        woodmac,
        _source_ref("train_store", "different-source"),
        ea,
        metadata,
    )
    assert not capacity._capacity_source_revision_is_coherent(
        _source_revision(),
        woodmac,
        train,
        ea,
        {},
    )


def test_capacity_scenario_revision_requires_selected_working_option_parity():
    revision, selected, working, options = _scenario_state()

    assert capacity._capacity_scenario_revision_is_coherent(
        revision,
        selected,
        working,
        {"dirty": False},
        "",
        options,
    )
    assert not capacity._capacity_scenario_revision_is_coherent(
        revision,
        selected + 1,
        working,
        {"dirty": False},
        "",
        options,
    )
    assert not capacity._capacity_scenario_revision_is_coherent(
        revision,
        selected,
        {**working, "base_revision": "different"},
        {"dirty": False},
        "",
        options,
    )
    assert not capacity._capacity_scenario_revision_is_coherent(
        revision,
        selected,
        working,
        {"dirty": False},
        "",
        [
            {
                **options[0],
                "current_snapshot_timestamp_utc": "different",
            }
        ],
    )


def test_empty_scenario_revision_clears_a_removed_selection():
    revision = {
        "format": "capacity_scenario_render_revision_v1",
        "source_key": SOURCE_KEY,
        "scenario_id": None,
        "empty": True,
        "dirty": False,
        "dirty_source": "",
        "dirty_updated_at": "",
        "refresh_revision": "removed",
    }

    assert capacity._capacity_scenario_revision_is_coherent(
        revision,
        None,
        None,
        {"dirty": False},
        "removed",
        [],
    )
    assert not capacity._capacity_scenario_revision_is_coherent(
        revision,
        None,
        {"format": capacity.CAPACITY_SCENARIO_WORKING_FORMAT},
        {"dirty": False},
        "removed",
        [],
    )


def test_mismatched_source_revision_stops_before_heavy_render(monkeypatch):
    monkeypatch.setattr(
        capacity,
        "_get_prepared_country_capacity_view",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("heavy render should not run")
        ),
    )

    with pytest.raises(PreventUpdate):
        capacity.render_capacity_table(
            _source_revision(),
            ["Qatar"],
            "exclude",
            "2026-01-01",
            "2026-12-31",
            "monthly",
            "detail",
            _source_ref("woodmac_store"),
            _source_ref("train_store", "different-source"),
            _source_ref("ea_store"),
            {"snapshot_key": SOURCE_KEY},
        )


def test_capacity_layout_and_callback_topology_use_two_revision_stores():
    component_ids = {
        getattr(component, "id", None)
        for component in _walk_capacity_components(capacity.layout)
    }
    assert "capacity-page-source-render-revision-store" in component_ids
    assert "capacity-page-scenario-render-revision-store" in component_ids

    callback_map = {
        **GLOBAL_CALLBACK_MAP,
        **index_shipping_snd.app.callback_map,
    }
    assert "capacity-page-source-render-revision-store.data" in callback_map
    assert "capacity-page-scenario-render-revision-store.data" in callback_map

    source_markers = (
        "capacity-page-woodmac-chart.figure",
        "capacity-page-ea-chart.figure",
        "capacity-page-unmapped-train-summary.children",
    )
    scenario_markers = (
        "capacity-page-yearly-capacity-comparison-chart.figure",
        "capacity-page-yearly-woodmac-capacity-discrepancy-table-container.children",
        "capacity-page-internal-scenario-chart.figure",
        "capacity-page-train-change-summary.children",
        "capacity-page-train-timeline-table-container.children",
    )
    lifecycle_ids = {
        "capacity-page-woodmac-data-store",
        "capacity-page-train-capacity-data-store",
        "capacity-page-ea-capacity-data-store",
        "capacity-page-metadata-store",
        "capacity-page-capacity-scenario-selected-store",
        "capacity-page-capacity-scenario-options-store",
        "capacity-page-capacity-scenario-working-store",
        "capacity-page-capacity-scenario-dirty-store",
        "capacity-page-capacity-scenario-refresh-store",
    }

    for callback_key, definition in callback_map.items():
        if any(marker in callback_key for marker in source_markers):
            input_ids = {
                dependency["id"]
                for dependency in definition["inputs"]
            }
            state_ids = {
                dependency["id"]
                for dependency in definition["state"]
            }
            assert "capacity-page-source-render-revision-store" in input_ids
            assert not input_ids.intersection(lifecycle_ids)
            assert state_ids.intersection(lifecycle_ids)
        if any(marker in callback_key for marker in scenario_markers):
            input_ids = {
                dependency["id"]
                for dependency in definition["inputs"]
            }
            state_ids = {
                dependency["id"]
                for dependency in definition["state"]
            }
            assert "capacity-page-scenario-render-revision-store" in input_ids
            assert not input_ids.intersection(lifecycle_ids)
            assert state_ids.intersection(lifecycle_ids)


def test_relational_capacity_metadata_carries_source_identity(monkeypatch):
    events = pd.DataFrame(
        {
            "provider_name": ["woodmac"],
            "country_name": ["Qatar"],
            "effective_month": pd.to_datetime(["2026-01-01"]),
        }
    )
    components = {
        "woodmac_store": events,
        "train_store": pd.DataFrame(),
        "ea_store": pd.DataFrame(),
    }
    monkeypatch.setattr(
        capacity,
        "_get_cached_capacity_source_snapshot",
        lambda _source_key: components,
    )
    monkeypatch.setattr(
        capacity,
        "fetch_capacity_source_state",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        capacity,
        "capacity_source_event_bounds",
        lambda _events: (
            pd.Timestamp("2026-01-01"),
            pd.Timestamp("2026-12-01"),
        ),
    )

    snapshot = capacity._fetch_relational_capacity_source_snapshot(
        SOURCE_KEY,
        {},
    )

    assert snapshot["snapshot_key"] == SOURCE_KEY
    assert snapshot["metadata"]["snapshot_key"] == SOURCE_KEY


# Consolidated from test_capacity_timeline_upload.py.

from io import BytesIO

import pandas as pd

from pages import capacity


def _scenario_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "scenario_row_key": "saved-event-1",
                "terminal_key": "terminal-1",
                "train_key": "train-1",
                "country_name": "Country A",
                "plant_name": "Plant A",
                "train_label": "Train 1 (GL1Z)",
                "base_provider": "woodmac",
                "base_first_date": "2028-01-01",
                "base_capacity_mtpa": 4.0,
                "scenario_first_date": "2029-01-01",
                "scenario_capacity_mtpa": 4.0,
                "scenario_note": "original note",
                "display_sort_plant": "Plant A",
                "display_sort_train": 1,
            },
            {
                "scenario_row_key": "saved-event-2",
                "terminal_key": "terminal-1",
                "train_key": "train-1",
                "country_name": "Country A",
                "plant_name": "Plant A",
                "train_label": "1",
                "base_provider": "internal_scenario",
                "base_first_date": "2030-01-01",
                "base_capacity_mtpa": 1.0,
                "scenario_first_date": "2030-01-01",
                "scenario_capacity_mtpa": 1.0,
                "scenario_note": "",
                "display_sort_plant": "Plant A",
                "display_sort_train": 1,
            },
        ]
    )


def _round_tripped_workbook(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    export_df = pd.DataFrame(
        {
            capacity.TRAIN_TIMELINE_IMPORT_KEY_COLUMN: rows["scenario_row_key"],
            "Country": rows["country_name"],
            "Plant": rows["plant_name"],
            "Train": rows["train_label"],
            "Scenario First Date": rows["scenario_first_date"],
            "Scenario Capacity": rows["scenario_capacity_mtpa"],
            "Scenario Note": rows["scenario_note"],
        }
    )
    metadata_df = capacity._build_train_timeline_upload_metadata_df(
        export_df,
        selected_scenario_id=99,
        scenario_name="Upload Test",
        original_name_visibility="hide",
        current_rows_df=rows,
    )
    workbook = capacity._export_train_timeline_workbook_bytes(
        export_df,
        metadata_df,
    )
    excel = pd.ExcelFile(BytesIO(workbook))
    return (
        pd.read_excel(excel, sheet_name=capacity.TRAIN_TIMELINE_SHEET_NAME, dtype=object),
        pd.read_excel(
            excel,
            sheet_name=capacity.TRAIN_TIMELINE_IMPORT_META_SHEET_NAME,
            dtype=object,
        ),
    )


def test_timeline_upload_restores_deleted_saved_row_with_canonical_identity():
    original_rows = _scenario_rows()
    main_df, metadata_df = _round_tripped_workbook(original_rows)

    unchanged, unchanged_summary = capacity._build_train_timeline_upload_rows_df(
        main_df,
        metadata_df,
        selected_scenario_id=99,
        current_rows_df=original_rows,
    )
    assert unchanged_summary == {"updated": 0, "added": 0, "deleted": 0}
    assert len(unchanged) == 2

    current_rows = original_rows[original_rows["scenario_row_key"] != "saved-event-1"]
    restored, restored_summary = capacity._build_train_timeline_upload_rows_df(
        main_df,
        metadata_df,
        selected_scenario_id=99,
        current_rows_df=current_rows,
    )

    assert restored_summary == {"updated": 0, "added": 1, "deleted": 0}
    restored_row = restored.set_index("scenario_row_key").loc["saved-event-1"]
    assert restored_row["train_key"] == "train-1"
    assert restored_row["terminal_key"] == "terminal-1"
    assert restored_row["base_provider"] == "woodmac"
    assert restored_row["base_capacity_mtpa"] == 4.0
    assert restored_row["scenario_capacity_mtpa"] == 4.0


# Consolidated from test_contracts_concurrent_loading.py.

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock
import json
import os
import subprocess
import sys
import time

import pandas as pd
import pytest

from pages import contracts
from utils import dashboard_snapshot_cache as snapshots


REVISION_V1 = (
    ("contracts", 1, "2026-07-27T00:00:00+00:00"),
    ("demand", 1, "2026-07-27T00:00:00+00:00"),
    ("price_assumptions", 1, "2026-07-27T00:00:00+00:00"),
    ("price_formula", 1, "2026-07-27T00:00:00+00:00"),
)
REVISION_V2 = tuple(
    (name, row_count, "2026-07-28T00:00:00+00:00")
    for name, row_count, _watermark in REVISION_V1
)


def _snapshot(revision_key=REVISION_V1, marker="v1"):
    return contracts.ContractsSnapshot(
        revision_key=revision_key,
        contracts=pd.DataFrame({"state": [f"contracts-{marker}"]}),
        demand=pd.DataFrame({"state": [f"demand-{marker}"]}),
        price_assumptions=pd.DataFrame(
            {"state": [f"assumptions-{marker}"]}
        ),
        price_formula=pd.DataFrame({"state": [f"formula-{marker}"]}),
        year_settings={"marker": marker},
    )


def _reset_contracts_state(monkeypatch):
    monkeypatch.setattr(contracts, "_contracts_snapshot", None)
    monkeypatch.setattr(
        contracts,
        "_contracts_snapshot_status",
        "unavailable",
    )


@pytest.fixture
def persistent_contracts_cache(monkeypatch, tmp_path):
    cache_directory = tmp_path / "contracts-cache"
    monkeypatch.setenv(snapshots.LOCAL_PERSISTENCE_ENV, "1")
    monkeypatch.setenv(
        snapshots.LOCAL_CACHE_DIR_ENV,
        str(cache_directory),
    )
    snapshots.close_persistent_snapshot_cache()
    snapshots.clear_local_snapshots()
    yield cache_directory
    snapshots.clear_local_snapshots()
    snapshots.close_persistent_snapshot_cache()
    monkeypatch.setattr(
        contracts,
        "_contracts_snapshot_message",
        "not loaded",
    )


def test_contract_loaders_overlap_and_assign_fixed_named_results(monkeypatch):
    loader_barrier = Barrier(4, timeout=2)
    expected = {
        "contracts": pd.DataFrame({"source": ["contracts"]}),
        "demand": pd.DataFrame({"source": ["demand"]}),
        "price_assumptions": pd.DataFrame(
            {"source": ["price_assumptions"]}
        ),
        "price_formula": pd.DataFrame({"source": ["price_formula"]}),
    }

    def loader(name):
        loader_barrier.wait()
        return expected[name]

    monkeypatch.setattr(
        contracts,
        "load_contracts_data",
        lambda: loader("contracts"),
    )
    monkeypatch.setattr(
        contracts,
        "load_annual_demand_data",
        lambda: loader("demand"),
    )
    monkeypatch.setattr(
        contracts,
        "load_price_assumptions_data",
        lambda: loader("price_assumptions"),
    )
    monkeypatch.setattr(
        contracts,
        "load_price_formula_data",
        lambda: loader("price_formula"),
    )
    monkeypatch.setattr(
        contracts,
        "_enhance_contracts_data",
        lambda contracts_df, assumptions_df: pd.DataFrame(
            {
                "contracts_source": contracts_df["source"],
                "assumptions_source": assumptions_df["source"],
            }
        ),
    )
    monkeypatch.setattr(
        contracts,
        "_calculate_contract_year_settings",
        lambda frame: {
            "rows": len(frame),
            "contracts_source": frame["contracts_source"].iloc[0],
        },
    )

    snapshot = contracts._build_contracts_snapshot(REVISION_V1)

    assert snapshot.revision_key == REVISION_V1
    assert snapshot.contracts.to_dict("records") == [
        {
            "contracts_source": "contracts",
            "assumptions_source": "price_assumptions",
        }
    ]
    pd.testing.assert_frame_equal(snapshot.demand, expected["demand"])
    pd.testing.assert_frame_equal(
        snapshot.price_assumptions,
        expected["price_assumptions"],
    )
    pd.testing.assert_frame_equal(
        snapshot.price_formula,
        expected["price_formula"],
    )
    assert snapshot.year_settings == {
        "rows": 1,
        "contracts_source": "contracts",
    }


def test_contract_snapshot_load_is_single_flight_across_callers(
    monkeypatch,
    persistent_contracts_cache,
):
    _reset_contracts_state(monkeypatch)
    call_count = 0
    count_lock = Lock()

    monkeypatch.setattr(
        contracts,
        "fetch_contracts_revision_key",
        lambda: REVISION_V1,
    )

    def build_snapshot(_revision_key):
        nonlocal call_count
        with count_lock:
            call_count += 1
        time.sleep(0.03)
        return _snapshot()

    monkeypatch.setattr(
        contracts,
        "_build_contracts_snapshot",
        build_snapshot,
    )

    with ThreadPoolExecutor(max_workers=6) as executor:
        results = list(
            executor.map(
                lambda _value: contracts._ensure_contracts_snapshot(),
                range(6),
            )
        )

    assert call_count == 1
    assert all(
        result.reference == results[0].reference
        and result.revision_key == results[0].revision_key
        for result in results
    )
    assert contracts._contracts_snapshot_status == "fresh"


def test_unchanged_revision_reuses_snapshot(
    monkeypatch,
    persistent_contracts_cache,
):
    _reset_contracts_state(monkeypatch)
    monkeypatch.setattr(
        contracts,
        "fetch_contracts_revision_key",
        lambda: REVISION_V1,
    )
    monkeypatch.setattr(
        contracts,
        "_build_contracts_snapshot",
        lambda _revision: _snapshot(),
    )
    existing = contracts._ensure_contracts_snapshot()
    monkeypatch.setattr(
        contracts,
        "_build_contracts_snapshot",
        lambda _revision: (_ for _ in ()).throw(
            AssertionError("unexpected reload")
        ),
    )

    result = contracts._ensure_contracts_snapshot()

    pd.testing.assert_frame_equal(result.contracts, existing.contracts)
    assert result.revision_key == existing.revision_key
    assert contracts._contracts_snapshot_status == "fresh"


def test_changed_revision_atomically_replaces_snapshot(
    monkeypatch,
    persistent_contracts_cache,
):
    existing = _snapshot()
    replacement = _snapshot(REVISION_V2, marker="v2")
    monkeypatch.setattr(contracts, "_contracts_snapshot", existing)
    monkeypatch.setattr(
        contracts,
        "fetch_contracts_revision_key",
        lambda: REVISION_V2,
    )
    monkeypatch.setattr(
        contracts,
        "_build_contracts_snapshot",
        lambda revision: replacement
        if revision == REVISION_V2
        else None,
    )

    result = contracts._ensure_contracts_snapshot()

    pd.testing.assert_frame_equal(result.contracts, replacement.contracts)
    assert snapshots.snapshot_is_resolvable(result.reference)
    assert contracts._contracts_snapshot is result
    assert contracts._contracts_snapshot_status == "fresh"


def test_forced_refresh_reloads_unchanged_revision(
    monkeypatch,
    persistent_contracts_cache,
):
    existing = _snapshot()
    replacement = _snapshot(marker="forced")
    monkeypatch.setattr(contracts, "_contracts_snapshot", existing)
    monkeypatch.setattr(
        contracts,
        "fetch_contracts_revision_key",
        lambda: REVISION_V1,
    )
    monkeypatch.setattr(
        contracts,
        "_build_contracts_snapshot",
        lambda _revision: replacement,
    )

    result = contracts._ensure_contracts_snapshot(force=True)

    pd.testing.assert_frame_equal(result.contracts, replacement.contracts)
    assert snapshots.snapshot_is_resolvable(result.reference)


def test_contract_refresh_failure_keeps_last_good_snapshot(
    monkeypatch,
    persistent_contracts_cache,
):
    existing = _snapshot()
    monkeypatch.setattr(contracts, "_contracts_snapshot", existing)
    monkeypatch.setattr(
        contracts,
        "fetch_contracts_revision_key",
        lambda: REVISION_V2,
    )
    monkeypatch.setattr(
        contracts,
        "_build_contracts_snapshot",
        lambda _revision: (_ for _ in ()).throw(
            RuntimeError("demand unavailable")
        ),
    )

    result = contracts._ensure_contracts_snapshot()

    pd.testing.assert_frame_equal(result.contracts, existing.contracts)
    assert result.revision_key == existing.revision_key
    assert contracts._contracts_snapshot is result
    assert result.status == "stale"
    assert contracts._contracts_snapshot_status == "stale"
    assert "last verified" in contracts._contracts_snapshot_message


def test_initial_contract_failure_returns_visible_unavailable_state(
    monkeypatch,
    persistent_contracts_cache,
):
    _reset_contracts_state(monkeypatch)
    monkeypatch.setattr(
        contracts,
        "fetch_contracts_revision_key",
        lambda: (_ for _ in ()).throw(RuntimeError("revision unavailable")),
    )

    result = contracts._ensure_contracts_snapshot()
    message, class_name = contracts._contracts_source_status(result)

    assert result.contracts.empty
    assert contracts._contracts_snapshot_status == "unavailable"
    assert "unavailable" in message
    assert class_name.endswith("-unavailable")


def test_contract_reference_survives_a_fresh_process_cache_state(
    monkeypatch,
    persistent_contracts_cache,
):
    _reset_contracts_state(monkeypatch)
    monkeypatch.setattr(
        contracts,
        "fetch_contracts_revision_key",
        lambda: REVISION_V1,
    )
    expected = _snapshot(marker="shared")
    monkeypatch.setattr(
        contracts,
        "_build_contracts_snapshot",
        lambda _revision: expected,
    )

    published = contracts._ensure_contracts_snapshot()
    reference = json.loads(json.dumps(published.reference))
    snapshots.clear_local_snapshots()
    snapshots.close_persistent_snapshot_cache()
    monkeypatch.setattr(contracts, "_contracts_snapshot", None)
    resolved = contracts._current_contracts_snapshot(reference)

    assert resolved.reference == reference
    assert resolved.revision_key == expected.revision_key
    assert resolved.year_settings == expected.year_settings
    for frame_name in (
        "contracts",
        "demand",
        "price_assumptions",
        "price_formula",
    ):
        pd.testing.assert_frame_equal(
            getattr(resolved, frame_name),
            getattr(expected, frame_name),
        )


@pytest.mark.parametrize("corruption_mode", ["missing", "corrupt"])
def test_contract_reference_never_falls_back_to_unrelated_global(
    monkeypatch,
    persistent_contracts_cache,
    corruption_mode,
):
    _reset_contracts_state(monkeypatch)
    monkeypatch.setattr(
        contracts,
        "fetch_contracts_revision_key",
        lambda: REVISION_V1,
    )
    monkeypatch.setattr(
        contracts,
        "_build_contracts_snapshot",
        lambda _revision: _snapshot(marker="published"),
    )
    reference = contracts._ensure_contracts_snapshot().reference
    stores = snapshots._get_persistent_stores()
    record_key = snapshots._disk_record_key(
        reference["namespace"],
        reference["source_key"],
        reference["revision"],
    )
    if corruption_mode == "missing":
        stores.cache.delete(record_key, retry=True)
    else:
        stores.cache.set(record_key, b"corrupt", retry=True)
    snapshots.clear_local_snapshots()
    monkeypatch.setattr(
        contracts,
        "_contracts_snapshot",
        _snapshot(REVISION_V2, marker="unrelated"),
    )

    with pytest.raises(
        snapshots.SnapshotUnavailable,
        match="Click the global Refresh",
    ):
        contracts._current_contracts_snapshot(reference)


def test_cross_worker_refresh_failure_keeps_exact_browser_reference(
    monkeypatch,
    persistent_contracts_cache,
):
    _reset_contracts_state(monkeypatch)
    monkeypatch.setattr(
        contracts,
        "fetch_contracts_revision_key",
        lambda: REVISION_V1,
    )
    monkeypatch.setattr(
        contracts,
        "_build_contracts_snapshot",
        lambda _revision: _snapshot(marker="last-good"),
    )
    reference = contracts._ensure_contracts_snapshot().reference
    snapshots.clear_local_snapshots()
    monkeypatch.setattr(contracts, "_contracts_snapshot", None)
    monkeypatch.setattr(
        contracts,
        "fetch_contracts_revision_key",
        lambda: REVISION_V2,
    )
    monkeypatch.setattr(
        contracts,
        "_build_contracts_snapshot",
        lambda _revision: (_ for _ in ()).throw(
            RuntimeError("source unavailable")
        ),
    )

    result = contracts._ensure_contracts_snapshot(
        force=True,
        fallback_reference=reference,
    )

    assert result.reference == reference
    assert result.year_settings == {"marker": "last-good"}
    assert contracts._contracts_snapshot_status == "stale"
    assert "last verified" in contracts._contracts_snapshot_message


def test_contract_reference_resolves_in_a_spawned_interpreter(
    monkeypatch,
    persistent_contracts_cache,
):
    _reset_contracts_state(monkeypatch)
    monkeypatch.setattr(
        contracts,
        "fetch_contracts_revision_key",
        lambda: REVISION_V1,
    )
    expected = _snapshot(marker="process-b")
    monkeypatch.setattr(
        contracts,
        "_build_contracts_snapshot",
        lambda _revision: expected,
    )
    reference = contracts._ensure_contracts_snapshot().reference
    environment = os.environ.copy()
    environment["CONTRACTS_TEST_REFERENCE"] = json.dumps(reference)
    script = """
import json
import os
from pages import contracts

reference = json.loads(os.environ["CONTRACTS_TEST_REFERENCE"])
snapshot = contracts._current_contracts_snapshot(reference)
print("CONTRACTS_RESULT=" + json.dumps({
    "contracts": snapshot.contracts.to_dict("records"),
    "demand": snapshot.demand.to_dict("records"),
    "price_assumptions": snapshot.price_assumptions.to_dict("records"),
    "price_formula": snapshot.price_formula.to_dict("records"),
    "revision_key": snapshot.revision_key,
    "year_settings": snapshot.year_settings,
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=os.path.dirname(os.path.dirname(__file__)),
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    result_line = next(
        line for line in completed.stdout.splitlines()
        if line.startswith("CONTRACTS_RESULT=")
    )
    result = json.loads(result_line.removeprefix("CONTRACTS_RESULT="))

    assert result["contracts"] == expected.contracts.to_dict("records")
    assert result["demand"] == expected.demand.to_dict("records")
    assert result["price_assumptions"] == (
        expected.price_assumptions.to_dict("records")
    )
    assert result["price_formula"] == (
        expected.price_formula.to_dict("records")
    )
    assert result["revision_key"] == json.loads(
        json.dumps(expected.revision_key)
    )
    assert result["year_settings"] == expected.year_settings


def test_evicted_matching_revision_is_republished_before_layout_reuse(
    monkeypatch,
    persistent_contracts_cache,
):
    _reset_contracts_state(monkeypatch)
    monkeypatch.setattr(
        contracts,
        "fetch_contracts_revision_key",
        lambda: REVISION_V1,
    )
    build_markers = iter(("first", "repaired"))
    monkeypatch.setattr(
        contracts,
        "_build_contracts_snapshot",
        lambda _revision: _snapshot(marker=next(build_markers)),
    )
    first = contracts._ensure_contracts_snapshot()
    stores = snapshots._get_persistent_stores()
    stores.cache.delete(
        snapshots._disk_record_key(
            first.reference["namespace"],
            first.reference["source_key"],
            first.reference["revision"],
        ),
        retry=True,
    )
    snapshots.clear_local_snapshots()

    repaired = contracts._ensure_contracts_snapshot()

    assert repaired.reference != first.reference
    assert repaired.year_settings == {"marker": "repaired"}
    assert contracts._current_contracts_snapshot(
        repaired.reference
    ).year_settings == {"marker": "repaired"}
    assert repaired.status == "fresh"


def test_none_or_malformed_refresh_fallback_is_unavailable_not_stale(
    monkeypatch,
    persistent_contracts_cache,
):
    _reset_contracts_state(monkeypatch)
    monkeypatch.setattr(
        contracts,
        "fetch_contracts_revision_key",
        lambda: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    none_result = contracts._ensure_contracts_snapshot(
        force=True,
        fallback_reference=None,
    )
    malformed_reference = {
        "format": snapshots.REFERENCE_FORMAT,
        "namespace": contracts.CONTRACTS_SNAPSHOT_NAMESPACE,
        "revision": "1" * 39,
        "shared": True,
    }
    malformed_result = contracts._ensure_contracts_snapshot(
        force=True,
        fallback_reference=malformed_reference,
    )

    assert none_result.status == "unavailable"
    assert malformed_result.status == "unavailable"
    assert none_result.reference is malformed_result.reference is None
    assert "unavailable" in none_result.message
    assert "unavailable" in malformed_result.message


def test_contract_consumers_show_visible_recovery_for_invalid_reference():
    invalid_reference = {
        "format": snapshots.REFERENCE_FORMAT,
        "namespace": contracts.CONTRACTS_SNAPSHOT_NAMESPACE,
        "revision": "1" * 39,
        "shared": True,
    }

    filter_options = contracts.update_filter_options(invalid_reference)
    assert len(filter_options) == 5
    assert all(
        options[0]["label"] == contracts.CONTRACTS_SNAPSHOT_RECOVERY_MESSAGE
        for options in filter_options
    )
    volume_outputs = contracts.update_volume_analysis_section(
        *([None] * 13),
        invalid_reference,
    )
    assert all(
        output.children == contracts.CONTRACTS_SNAPSHOT_RECOVERY_MESSAGE
        for output in volume_outputs
    )
    timeline_outputs = contracts.update_timeline_charts(
        *([None] * 7),
        invalid_reference,
    )
    assert all(
        figure.layout.annotations[0].text
        == contracts.CONTRACTS_SNAPSHOT_RECOVERY_MESSAGE
        for figure in timeline_outputs
    )
    columns, rows = contracts.update_contracts_table(
        *([None] * 7),
        invalid_reference,
    )
    assert columns == [{"headerName": "Status", "field": "snapshot_status"}]
    assert rows == [{
        "snapshot_status": contracts.CONTRACTS_SNAPSHOT_RECOVERY_MESSAGE,
    }]


def test_contract_source_drift_is_not_published(
    monkeypatch,
    persistent_contracts_cache,
):
    _reset_contracts_state(monkeypatch)
    revisions = iter((REVISION_V1, REVISION_V2))
    monkeypatch.setattr(
        contracts,
        "fetch_contracts_revision_key",
        lambda: next(revisions),
    )
    monkeypatch.setattr(
        contracts,
        "_build_contracts_snapshot",
        lambda _revision: _snapshot(marker="mixed"),
    )

    result = contracts._ensure_contracts_snapshot()

    assert result.status == "unavailable"
    assert snapshots.get_snapshot_if_available(
        contracts.engine,
        namespace=contracts.CONTRACTS_SNAPSHOT_NAMESPACE,
        source_key=contracts._contracts_snapshot_source_key(REVISION_V1),
    ) is None


# Consolidated from test_database_singleton.py.

from concurrent.futures import ThreadPoolExecutor
import json
import os
import subprocess
import sys
from threading import Event

from pages import (
    contracts,
    country_mappings,
    exporter_detail,
    exporters,
    fleet_metrics,
    importer_detail,
    terminal_adjustments,
    train_names_mapping,
)
from utils import database, export_flow_data, import_flow_data


def test_active_modules_share_one_database_engine():
    modules = (
        contracts,
        country_mappings,
        exporter_detail,
        exporters,
        fleet_metrics,
        importer_detail,
        terminal_adjustments,
        train_names_mapping,
        export_flow_data,
        import_flow_data,
    )

    assert {id(module.engine) for module in modules} == {
        id(database.engine)
    }


def test_database_pool_defaults_are_bounded():
    pool = database.engine.pool

    assert pool.size() == database.DEFAULT_POOL_SIZE == 2
    assert pool._max_overflow == database.DEFAULT_MAX_OVERFLOW == 0
    assert pool._timeout == database.DEFAULT_POOL_TIMEOUT_SECONDS == 30
    assert pool._recycle == database.DEFAULT_POOL_RECYCLE_SECONDS == 1800
    with database.engine.connect() as connection:
        application_name = connection.exec_driver_sql(
            "SELECT current_setting('application_name')"
        ).scalar_one()
    assert application_name == os.getenv(
        "DASH_DB_APPLICATION_NAME",
        database.DEFAULT_APPLICATION_NAME,
    )


def test_country_geometry_is_pinned_cached_and_copied(monkeypatch):
    from pages import country_mappings
    import requests

    payload = {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "properties": {}}],
    }
    calls = []

    class _Response:
        def raise_for_status(self):
            calls.append("status")

        def json(self):
            return payload

    def get(url, *, timeout):
        calls.append((url, timeout))
        return _Response()

    monkeypatch.setattr(requests, "get", get)
    country_mappings._fetch_world_geojson.cache_clear()
    first = country_mappings._world_geojson_copy()
    second = country_mappings._world_geojson_copy()
    first["features"][0]["properties"]["changed"] = True

    assert calls == [
        (country_mappings.WORLD_GEOJSON_URL, 10),
        "status",
    ]
    assert "ca96624a56bd078437bca8184e78163e5039ad19" in (
        country_mappings.WORLD_GEOJSON_URL
    )
    assert second == payload
    country_mappings._fetch_world_geojson.cache_clear()


def test_country_geometry_cold_load_is_single_flight(monkeypatch):
    from pages import country_mappings
    import requests

    payload = {"type": "FeatureCollection", "features": []}
    request_started = Event()
    release_request = Event()
    request_count = 0

    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return payload

    def get(_url, *, timeout):
        nonlocal request_count
        assert timeout == 10
        request_count += 1
        request_started.set()
        assert release_request.wait(timeout=2)
        return _Response()

    monkeypatch.setattr(requests, "get", get)
    country_mappings._fetch_world_geojson.cache_clear()
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(country_mappings._world_geojson_copy)
        assert request_started.wait(timeout=2)
        second = executor.submit(country_mappings._world_geojson_copy)
        release_request.set()
        assert first.result(timeout=2) == payload
        assert second.result(timeout=2) == payload

    assert request_count == 1
    country_mappings._fetch_world_geojson.cache_clear()


def test_nondefault_schema_reaches_all_previously_hardcoded_sql(tmp_path):
    config_path = tmp_path / "config.ini"
    config_path.write_text(
        "[DATABASE]\n"
        "CONNECTION_STRING = postgresql+psycopg2://u:p@127.0.0.1:9/db\n"
        "SCHEMA = alternate_lng\n"
    )
    environment = os.environ.copy()
    environment["DASHBOARD_CONFIG_FILE"] = str(config_path)
    script = """
import json
import pandas as pd
from pages import contracts, exporter_detail, importer_detail, shipping_balance
from utils import dashboard_snapshot_cache as snapshots

queries = []
contracts.pd.read_sql = lambda query, _engine: (
    queries.append(str(query)) or pd.DataFrame()
)
contracts.load_contracts_data()
contracts.load_annual_demand_data()
contracts.load_price_assumptions_data()
contracts.load_price_formula_data()
values = queries + [
    str(contracts.CONTRACTS_REVISION_QUERY),
    shipping_balance.SHIPPING_BALANCE_SOURCE_STATE_QUERY,
    shipping_balance.SHIPPING_BALANCE_DATE_STATE_QUERY,
    exporter_detail.WOODMAC_IMPORT_EXPORTS_TABLE,
    importer_detail.WOODMAC_IMPORT_EXPORTS_TABLE,
    snapshots.SNAPSHOT_TABLE,
]
print("SCHEMA_RESULT=" + json.dumps(values))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=os.path.dirname(os.path.dirname(__file__)),
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    result_line = next(
        line for line in completed.stdout.splitlines()
        if line.startswith("SCHEMA_RESULT=")
    )

    assert all(
        "alternate_lng." in value
        for value in json.loads(
            result_line.removeprefix("SCHEMA_RESULT=")
        )
    )


# Consolidated from test_page_registry.py.

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


# Consolidated from test_shipping_balance_deactivation.py.

import json
from pathlib import Path
import subprocess
import sys
import textwrap


_SHIPPING_REPO_ROOT = Path(__file__).resolve().parents[1]


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
        cwd=_SHIPPING_REPO_ROOT,
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
    route_source = (_SHIPPING_REPO_ROOT / "index_shipping_snd.py").read_text()
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


# Consolidated from test_terminal_adjustments_concurrent_layout.py.

from threading import Barrier

import pandas as pd
import pytest

from pages import terminal_adjustments


def _walk_terminal_components(component):
    yield component
    children = getattr(component, "children", None)
    if children is None:
        return
    if not isinstance(children, (list, tuple)):
        children = [children]
    for child in children:
        if hasattr(child, "to_plotly_json"):
            yield from _walk_terminal_components(child)


def _components_by_id(layout):
    return {
        component.id: component
        for component in _walk_terminal_components(layout)
        if getattr(component, "id", None)
    }


def test_layout_loaders_overlap_and_preserve_exact_options(monkeypatch):
    loader_barrier = Barrier(3, timeout=2)
    scenarios = ["base_view", "best_view", "stress_case"]
    plants = pd.DataFrame(
        [
            {"plant_name": "Alpha", "country_name": "Country A"},
            {"plant_name": "Beta", "country_name": "Country B"},
        ]
    )
    trains = pd.DataFrame(
        [
            {
                "plant_name": "Alpha",
                "lng_train_name_short": "T1",
                "country_name": "Country A",
                "id_plant": 1,
                "id_lng_train": 10,
            },
            {
                "plant_name": "Beta",
                "lng_train_name_short": "T2",
                "country_name": "Country B",
                "id_plant": None,
                "id_lng_train": None,
            },
        ]
    )

    def load(value):
        loader_barrier.wait()
        return value

    monkeypatch.setattr(
        terminal_adjustments,
        "get_available_scenarios",
        lambda _engine: load(scenarios),
    )
    monkeypatch.setattr(
        terminal_adjustments,
        "get_plants_list",
        lambda: load(plants),
    )
    monkeypatch.setattr(
        terminal_adjustments,
        "get_trains_list",
        lambda: load(trains),
    )

    components = _components_by_id(terminal_adjustments.layout())

    assert components["scenario-selector"].options == [
        {"label": "base_view", "value": "base_view"},
        {"label": "best_view", "value": "best_view"},
        {"label": "stress_case", "value": "stress_case"},
    ]
    assert components["scenario-selector"].value == "best_view"
    assert components["plant-filter"].options == [
        {"label": "Alpha (Country A)", "value": "Alpha"},
        {"label": "Beta (Country B)", "value": "Beta"},
    ]
    assert components["train-filter"].options == [
        {"label": "Alpha - T1", "value": "Alpha|T1"},
        {"label": "Beta - T2", "value": "Beta|T2"},
    ]
    assert components["trains-to-copy"].options == [
        {"label": "Alpha - T1", "value": "Alpha|T1"},
    ]
    assert [
        child.value
        for child in components["scenario-list"].children
    ] == ["best_view", "stress_case"]


def test_layout_propagates_option_loader_failure(monkeypatch):
    monkeypatch.setattr(
        terminal_adjustments,
        "get_available_scenarios",
        lambda _engine: ["base_view", "best_view"],
    )
    monkeypatch.setattr(
        terminal_adjustments,
        "get_plants_list",
        lambda: (_ for _ in ()).throw(RuntimeError("plants unavailable")),
    )
    monkeypatch.setattr(
        terminal_adjustments,
        "get_trains_list",
        lambda: pd.DataFrame(),
    )

    with pytest.raises(RuntimeError, match="plants unavailable"):
        terminal_adjustments.layout()


def test_copy_trains_binds_allowlisted_browser_values(monkeypatch):
    plant_name = "O'Brien' OR 1=1 --"
    train_name = "Train A"
    monkeypatch.setattr(
        terminal_adjustments,
        "get_trains_list",
        lambda: pd.DataFrame(
            [{
                "plant_name": plant_name,
                "lng_train_name_short": train_name,
                "id_plant": 7,
                "id_lng_train": 11,
            }]
        ),
    )

    class _Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class _Engine:
        def connect(self):
            return _Connection()

    fake_engine = _Engine()
    monkeypatch.setattr(terminal_adjustments, "engine", fake_engine)
    captured = {}

    def read_sql(statement, _connection, params=None):
        captured["statement"] = str(statement)
        captured["params"] = params
        return pd.DataFrame(
            [{
                "id_plant": 7,
                "id_lng_train": 11,
                "plant_name": plant_name,
                "lng_train_name_short": train_name,
                "year": 2026,
                "month": 1,
                "baseline_output": 2.5,
            }]
        )

    monkeypatch.setattr(terminal_adjustments.pd, "read_sql", read_sql)
    monkeypatch.setattr(
        terminal_adjustments,
        "save_adjustments_bulk",
        lambda frame, scenario, selected_engine: (
            captured.update(
                frame=frame.copy(),
                scenario=scenario,
                engine=selected_engine,
            )
            or {"records_inserted": len(frame)}
        ),
    )

    message, class_name = terminal_adjustments.copy_trains_to_scenario(
        1,
        [f"{plant_name}|{train_name}"],
        "best_view",
    )

    assert plant_name not in captured["statement"]
    assert train_name not in captured["statement"]
    assert captured["params"] == {
        "plant_name_0": plant_name,
        "train_name_0": train_name,
    }
    assert captured["frame"]["adjusted_output"].tolist() == [2.5]
    assert captured["scenario"] == "best_view"
    assert captured["engine"] is fake_engine
    assert message == (
        "Successfully copied 1 records for 1 train(s) to 'best_view'"
    )
    assert class_name == "alert alert-success"


@pytest.mark.parametrize(
    "selection",
    [
        "Alpha|T1",
        [42],
        ["Alpha"],
        ["|T1"],
        ["Unknown|T9"],
        ["Alpha|T1", "Alpha|T1"],
    ],
)
def test_copy_trains_rejects_forged_enumerations_before_sql(
    monkeypatch,
    selection,
):
    monkeypatch.setattr(
        terminal_adjustments,
        "get_trains_list",
        lambda: pd.DataFrame(
            [{
                "plant_name": "Alpha",
                "lng_train_name_short": "T1",
                "id_plant": 1,
                "id_lng_train": 10,
            }]
        ),
    )
    monkeypatch.setattr(
        terminal_adjustments.pd,
        "read_sql",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("SQL must not run")
        ),
    )
    monkeypatch.setattr(
        terminal_adjustments,
        "save_adjustments_bulk",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("write must not run")
        ),
    )

    message, class_name = terminal_adjustments.copy_trains_to_scenario(
        1,
        selection,
        "best_view",
    )

    assert message == "Error copying trains: Invalid train selection"
    assert class_name == "alert alert-danger"
