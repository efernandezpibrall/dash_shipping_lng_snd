import copy
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest
from sqlalchemy import create_engine, text

from pages import exporter_detail, importer_detail
from utils import dashboard_snapshot_cache as snapshots
from utils import detail_snapshot_precompute as precompute


@pytest.fixture
def persistent_precompute_cache(monkeypatch, tmp_path):
    cache_directory = tmp_path / "detail-precompute-cache"
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


def _source_watermark():
    return {
        "kpler_watermark": "2026-07-27T00:00:00Z",
        "woodmac_watermark": "2026-07-26T00:00:00Z",
        "distance_watermark": "2026-07-25T00:00:00Z",
        "mapping_fingerprint": "mapping-v1",
    }


def _maintenance_version():
    return {
        "planned_watermark": "2026-07-26T00:00:00Z",
        "planned_row_count": 4,
        "unplanned_watermark": "2026-07-26T00:00:00Z",
        "unplanned_row_count": 3,
        "train_capacity_watermark": "2026-07-26T00:00:00Z",
        "train_capacity_row_count": 2,
        "plant_capacity_watermark": "2026-07-26T00:00:00Z",
        "plant_capacity_row_count": 1,
        "mapping_fingerprint": "mapping-v1",
    }


def _allocation_run():
    return {
        "run_id": "allocation-run-v1",
        "analysis_date": pd.Timestamp("2026-07-27T00:00:00"),
        "forecast_start": pd.Timestamp("2026-07-01"),
        "forecast_end": pd.Timestamp("2028-12-01"),
    }


def _importer_catalog():
    return [
        {
            "destination_country_name": "China",
            "country": "China",
            "country_display": "China",
            "continent": "Asia",
            "subcontinent": "East Asia",
            "basin": "Pacific",
            "country_classification_level1": "Asia",
            "country_classification": "Importer",
            "shipping_region": "North Asia",
        }
    ]


def _payload(section, selection):
    return {
        "section": section,
        "selection": selection,
        "frame": pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2026-01-01", "2026-02-01", None]
                ),
                "value": [1.25, None, 3.5],
            }
        ),
    }


def _patch_exporter_sources(monkeypatch, *, fail_builders=False):
    monkeypatch.setattr(
        exporter_detail,
        "_fetch_exporter_detail_source_watermark",
        lambda: copy.deepcopy(_source_watermark()),
    )
    monkeypatch.setattr(
        exporter_detail,
        "_fetch_exporter_maintenance_source_version",
        lambda: copy.deepcopy(_maintenance_version()),
    )
    monkeypatch.setattr(
        exporter_detail,
        "fetch_latest_supply_allocation_run_metadata",
        lambda _engine: copy.deepcopy(_allocation_run()),
    )
    monkeypatch.setattr(
        exporter_detail,
        "_fetch_exporter_diversion_source_version",
        lambda: {"diversion_watermark": "diversion-v1"},
    )

    def builder(section):
        if fail_builders:
            return lambda *_args, **_kwargs: pytest.fail(
                f"{section} exporter builder should not run"
            )
        return lambda *_args, **_kwargs: _payload(
            section,
            "United States",
        )

    monkeypatch.setattr(
        exporter_detail,
        "_build_exporter_detail_base_payload",
        builder("base"),
    )
    monkeypatch.setattr(
        exporter_detail,
        "_fetch_destination_forecast_source_data",
        builder("allocation"),
    )
    monkeypatch.setattr(
        exporter_detail,
        "_build_exporter_maintenance_source_payload",
        builder("maintenance"),
    )
    monkeypatch.setattr(
        exporter_detail,
        "_build_exporter_route_source_payload",
        builder("route"),
    )
    monkeypatch.setattr(
        exporter_detail,
        "_build_exporter_diversion_payload",
        builder("diversion"),
    )


def _patch_importer_sources(monkeypatch, *, fail_builders=False):
    monkeypatch.setattr(
        importer_detail,
        "build_destination_catalog",
        lambda _engine: copy.deepcopy(_importer_catalog()),
    )
    monkeypatch.setattr(
        importer_detail,
        "_fetch_importer_detail_source_watermark",
        lambda: copy.deepcopy(_source_watermark()),
    )
    monkeypatch.setattr(
        importer_detail,
        "_fetch_importer_maintenance_source_version",
        lambda: copy.deepcopy(_maintenance_version()),
    )
    monkeypatch.setattr(
        importer_detail,
        "fetch_latest_supply_allocation_run_metadata",
        lambda _engine: copy.deepcopy(_allocation_run()),
    )
    monkeypatch.setattr(
        importer_detail,
        "_fetch_importer_diversion_source_version",
        lambda: {"diversion_watermark": "diversion-v1"},
    )

    def builder(section):
        if fail_builders:
            return lambda *_args, **_kwargs: pytest.fail(
                f"{section} importer builder should not run"
            )
        return lambda *_args, **_kwargs: _payload(section, "China")

    monkeypatch.setattr(
        importer_detail,
        "_build_import_analysis_base_payload",
        builder("base"),
    )
    monkeypatch.setattr(
        importer_detail,
        "_fetch_origin_forecast_source_data",
        builder("allocation"),
    )
    monkeypatch.setattr(
        importer_detail,
        "_build_importer_maintenance_source_payload",
        builder("maintenance"),
    )
    monkeypatch.setattr(
        importer_detail,
        "_build_importer_route_source_payload",
        builder("route"),
    )
    monkeypatch.setattr(
        importer_detail,
        "_build_importer_diversion_payload",
        builder("diversion"),
    )


def _summary_references(summary):
    return {
        (item["page"], item["section"]): (
            item["namespace"],
            item["source_key"],
            item["revision"],
        )
        for item in summary["snapshots"]
    }


def _reference_identity(reference):
    return (
        reference["namespace"],
        reference["source_key"],
        reference["revision"],
    )


def test_precompute_survives_cache_reopen_and_normal_loaders_do_not_rebuild(
    monkeypatch,
    persistent_precompute_cache,
):
    _patch_exporter_sources(monkeypatch)
    _patch_importer_sources(monkeypatch)

    summary = precompute.precompute_detail_snapshots(
        exporter_detail,
        importer_detail,
        exporter_countries=["United States"],
        importer_targets=[("Country", "China")],
    )

    assert summary["status"] == "ready"
    assert len(summary["snapshots"]) == 10
    assert summary["sql_audit"]["rejected_statement_count"] == 0
    assert summary["cache"]["volume_bytes"] > 0
    expected = _summary_references(summary)

    # Simulate a fresh Dash worker: only persistent records may satisfy these
    # normal page-loader calls.
    snapshots.clear_local_snapshots()
    snapshots.close_persistent_snapshot_cache()
    _patch_exporter_sources(monkeypatch, fail_builders=True)
    _patch_importer_sources(monkeypatch, fail_builders=True)

    exporter_context = precompute.build_exporter_source_context(
        exporter_detail
    )
    importer_context = precompute.build_importer_source_context(
        importer_detail
    )
    catalog = _importer_catalog()

    exporter_refs = {
        "base": exporter_detail.refresh_exporter_detail_base_data(
            "United States",
            source_context=exporter_context,
        ),
        "allocation": exporter_detail.refresh_destination_forecast_source(
            "United States",
            exporter_context,
        ),
        "maintenance": (
            exporter_detail._get_exporter_maintenance_source_reference(
                "United States",
                exporter_context,
            )[0]
        ),
        "route": exporter_detail.refresh_exporter_route_analysis_source(
            "United States",
            source_context=exporter_context,
        ),
        "diversion": exporter_detail.refresh_exporter_diversion_source(
            exporter_context,
            "United States",
        ),
    }
    importer_refs = {
        "base": importer_detail.refresh_import_analysis_base_data(
            0,
            "country",
            "China",
            catalog,
            source_context=importer_context,
        ),
        "allocation": importer_detail.refresh_origin_forecast_source(
            importer_context,
            "country",
            "China",
            catalog,
        ),
        "maintenance": (
            importer_detail._get_importer_maintenance_source_reference(
                ["China"],
                importer_context,
            )[0]
        ),
        "route": importer_detail.refresh_importer_route_analysis_source(
            "country",
            "China",
            catalog,
            importer_context,
        ),
        "diversion": importer_detail.refresh_importer_diversion_source(
            importer_context,
            "country",
            "China",
            catalog,
        ),
    }

    for section, reference in exporter_refs.items():
        assert _reference_identity(reference) == expected[
            ("exporter", section)
        ]
        payload = snapshots.resolve_snapshot(
            reference,
            exporter_detail.engine,
        )
        manifest = snapshots.resolve_snapshot_manifest(
            reference,
            exporter_detail.engine,
        )
        assert isinstance(payload["frame"], pd.DataFrame)
        assert manifest

    for section, reference in importer_refs.items():
        assert _reference_identity(reference) == expected[
            ("importer", section)
        ]
        payload = snapshots.resolve_snapshot(
            reference,
            importer_detail.engine,
        )
        manifest = snapshots.resolve_snapshot_manifest(
            reference,
            importer_detail.engine,
        )
        assert isinstance(payload["frame"], pd.DataFrame)
        assert manifest


def test_precompute_fails_closed_when_local_persistence_is_disabled(
    monkeypatch,
):
    monkeypatch.setenv(snapshots.LOCAL_PERSISTENCE_ENV, "0")

    with pytest.raises(
        precompute.DetailSnapshotPrecomputeError,
        match="legacy Postgres backend",
    ):
        precompute.precompute_detail_snapshots(
            exporter_detail,
            importer_detail,
            exporter_countries=[],
            importer_targets=[],
        )


def test_sql_audit_allows_reads_and_rejects_ddl():
    engine = create_engine("sqlite:///:memory:")
    try:
        with precompute.audit_read_only_sql([engine]) as audit:
            with engine.connect() as connection:
                assert connection.execute(text("SELECT 1")).scalar() == 1
                with pytest.raises(
                    precompute.DetailSnapshotPrecomputeError,
                    match="CREATE",
                ):
                    connection.execute(
                        text("CREATE TABLE forbidden (id INTEGER)")
                    )
        assert audit.statement_count == 2
        assert audit.rejected_statement_count == 1
    finally:
        engine.dispose()


def test_cli_help_does_not_import_dashboard_pages():
    repository_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(
                repository_root
                / "scripts"
                / "precompute_detail_snapshots.py"
            ),
            "--help",
        ],
        cwd=repository_root,
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "country=China" in result.stdout
