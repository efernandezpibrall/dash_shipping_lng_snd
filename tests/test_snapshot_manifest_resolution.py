import pytest

from utils import dashboard_snapshot_cache as snapshots


def test_exact_manifest_survives_restart_and_newer_revision(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv(snapshots.LOCAL_PERSISTENCE_ENV, "1")
    monkeypatch.setenv(
        snapshots.LOCAL_CACHE_DIR_ENV,
        str(tmp_path / "manifest-cache"),
    )
    snapshots.close_persistent_snapshot_cache()
    snapshots.clear_local_snapshots()

    namespace = "provider-flow-source-v2"
    source_key = "provider-source"
    old_state = {
        "mapping_hash": "country-v1",
        "ea_balance_mapping_hash": "ea-v1",
    }
    old_reference, _ = snapshots.get_or_build_snapshot(
        None,
        namespace=namespace,
        source_key=source_key,
        builder=lambda: {"value": "old"},
        manifest={"source_state": old_state},
    )

    snapshots.clear_local_snapshots()
    snapshots.close_persistent_snapshot_cache()
    assert snapshots.resolve_snapshot_manifest(
        old_reference,
        None,
        expected_namespace=namespace,
    ) == {"source_state": old_state}

    new_state = {
        "mapping_hash": "country-v2",
        "ea_balance_mapping_hash": "ea-v2",
    }
    new_reference, _ = snapshots.get_or_build_snapshot(
        None,
        namespace=namespace,
        source_key=source_key,
        builder=lambda: {"value": "new"},
        manifest={"source_state": new_state},
        force=True,
    )
    assert new_reference != old_reference
    assert snapshots.resolve_snapshot_manifest(
        old_reference,
        None,
        expected_namespace=namespace,
    ) == {"source_state": old_state}
    assert snapshots.resolve_snapshot_manifest(
        new_reference,
        None,
        expected_namespace=namespace,
    ) == {"source_state": new_state}

    latest_reference, latest_payload = snapshots.get_or_build_snapshot(
        None,
        namespace=namespace,
        source_key=source_key,
        builder=lambda: {"value": "unexpected"},
    )
    assert latest_reference == new_reference
    assert latest_payload == {"value": "new"}

    with pytest.raises(snapshots.SnapshotUnavailable):
        snapshots.resolve_snapshot_manifest(
            old_reference,
            None,
            expected_namespace="wrong-namespace",
        )
    with pytest.raises(snapshots.SnapshotUnavailable):
        snapshots.resolve_snapshot_manifest(
            {**old_reference, "revision": "corrupt"},
            None,
            expected_namespace=namespace,
        )

    snapshots.clear_local_snapshots()
    snapshots.close_persistent_snapshot_cache()
