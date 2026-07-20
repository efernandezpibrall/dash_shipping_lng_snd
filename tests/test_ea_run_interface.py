import pytest

from utils import ea_run_interface
from utils import export_flow_data
from utils import import_flow_data
from utils import market_balance_data
from utils import provider_flow_snapshot
from utils import snapshot_controls


@pytest.mark.parametrize(
    "value",
    [None, True, False, 0, -1, 1.0, "1", "2026-07-16T00:00:00"],
)
def test_ea_run_id_rejects_implicit_coercions(value):
    with pytest.raises(ValueError, match="positive integer"):
        ea_run_interface.normalize_ea_run_id(value)


def test_ea_run_dropdown_uses_integer_values_and_disambiguates_timestamps():
    runs = [
        {
            "run_id": 12,
            "snapshot_at": "2026-07-16T10:00:00Z",
            "change_count": 1,
            "delete_count": 0,
        },
        {
            "run_id": 11,
            "snapshot_at": "2026-07-16T10:00:00Z",
            "change_count": 2,
            "delete_count": 1,
        },
        {
            "run_id": 10,
            "snapshot_at": "2026-07-15T10:00:00Z",
            "change_count": 3,
            "delete_count": 0,
        },
    ]

    options = snapshot_controls.build_ea_upload_dropdown_options(runs)

    assert [option["value"] for option in options] == [12, 11, 10]
    assert options[0]["label"].endswith("run 12")
    assert options[1]["label"].endswith("run 11")
    assert options[2]["label"] == "2026-07-15 10:00"


def test_hot_browser_timestamp_state_resets_to_second_changed_run():
    comparison_options = {
        "woodmac": {"short_term": [], "long_term": []},
        "ea_comparison_runs": [
            {"run_id": 12, "snapshot_at": "2026-07-16T10:00:00Z"},
            {"run_id": 11, "snapshot_at": "2026-07-15T10:00:00Z"},
        ],
    }

    resolved = snapshot_controls.resolve_snapshot_control_values(
        "ea",
        comparison_options,
        None,
        None,
        "2026-07-15T10:00:00Z",
    )

    assert resolved[5] == 11
    assert isinstance(resolved[4][0]["value"], int)


def test_flow_queries_use_only_public_ea_interfaces_and_bound_function_inputs():
    current_queries = (
        export_flow_data._build_ea_export_flow_query(),
        import_flow_data._build_ea_import_flow_query(),
    )
    historical_queries = (
        export_flow_data._build_ea_parameterized_export_flow_query(),
        import_flow_data._build_ea_parameterized_import_flow_query(),
    )

    for query in current_queries:
        assert ".ea_values_current" in query
        assert ".ea_values " not in query
    for query in historical_queries:
        assert "ea_values_at_run" in query
        assert ":ea_as_of_run_id" in query
        assert ":ea_start_date" in query
        assert ":ea_end_date" in query
        assert "array_agg(" in query
        assert "HAVING count(*) > 0" in query
        assert "CROSS JOIN LATERAL" in query
        assert ".ea_values " not in query


def test_current_metadata_is_not_derived_from_changed_comparison_runs():
    metadata = snapshot_controls.ea_metadata_from_upload_options(
        {"run_id": 20, "snapshot_at": "2026-07-16T12:00:00Z"}
    )

    assert metadata == {
        "run_id": 20,
        "snapshot_at": "2026-07-16T12:00:00Z",
        "upload_timestamp_utc": "2026-07-16T12:00:00Z",
    }


def test_captured_provider_state_fails_closed_if_sources_change(monkeypatch):
    captured = {
        "current_ea": {"run_id": 20, "snapshot_at": "2026-07-16T12:00:00Z"},
        "mapping_hash": "before",
    }
    monkeypatch.setattr(
        provider_flow_snapshot,
        "get_or_build_snapshot",
        lambda *args, **kwargs: ("reference", {"current_ea": captured["current_ea"]}),
    )
    monkeypatch.setattr(
        provider_flow_snapshot,
        "fetch_provider_flow_source_state",
        lambda: {**captured, "mapping_hash": "after"},
    )

    with pytest.raises(RuntimeError, match="captured page state"):
        provider_flow_snapshot.get_provider_flow_snapshot_for_state(captured)


def test_provider_source_state_hashes_effective_ea_catalog_and_selection(monkeypatch):
    executed_sql = []

    class Result:
        def scalar(self):
            return "mapping-revision"

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, statement):
            executed_sql.append(str(statement))
            return Result()

    class Engine:
        def connect(self):
            return Connection()

    monkeypatch.setattr(provider_flow_snapshot, "engine", Engine())
    monkeypatch.setattr(
        provider_flow_snapshot,
        "build_resolved_ea_lng_balance_ctes",
        lambda *_args: (
            "resolved AS (SELECT '1'::text AS dataset_id, "
            "''::text AS country, ''::text AS country_iso, ''::text AS region, "
            "''::text AS sub_region, ''::text AS description, ''::text AS aspect, "
            "''::text AS aspect_subtype, ''::text AS category, "
            "''::text AS category_subtype, ''::text AS frequency, "
            "''::text AS lifecycle_stage, ''::text AS source, ''::text AS unit)",
            "resolved",
        ),
    )

    assert provider_flow_snapshot._fetch_ea_balance_mapping_hash() == "mapping-revision"
    rendered = executed_sql[-1]
    for field in ("dataset_id", "country", "aspect", "category_subtype", "frequency", "unit"):
        assert field in rendered


def test_provider_source_key_carries_the_effective_ea_mapping_hash(monkeypatch):
    class Result:
        def scalar(self):
            return "country-revision"

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, _statement):
            return Result()

    class Engine:
        def connect(self):
            return Connection()

    monkeypatch.setattr(provider_flow_snapshot, "engine", Engine())
    monkeypatch.setattr(
        provider_flow_snapshot, "_SOURCE_STATE_QUERIES", {"mapping_hash": object()}
    )
    monkeypatch.setattr(
        provider_flow_snapshot, "_fetch_ea_balance_mapping_hash", lambda: "ea-revision"
    )
    monkeypatch.setattr(
        provider_flow_snapshot,
        "fetch_current_ea_run",
        lambda *_args, **_kwargs: {"run_id": 42, "snapshot_at": "2026-07-16T00:00:00Z"},
    )

    state = provider_flow_snapshot.fetch_provider_flow_source_state()

    assert state["ea_balance_mapping_hash"] == "ea-revision"
    assert state["current_ea"]["run_id"] == 42


def test_pinned_market_payload_does_not_fallback_to_moving_latest(monkeypatch):
    captured = {"current_ea": {"run_id": 20}}
    monkeypatch.setattr(
        market_balance_data,
        "get_provider_flow_snapshot_for_state",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("state changed")),
    )
    fallback_called = False

    def fallback():
        nonlocal fallback_called
        fallback_called = True
        return {}, {}

    monkeypatch.setattr(market_balance_data, "build_provider_flow_payload", fallback)
    with pytest.raises(RuntimeError, match="state changed"):
        market_balance_data._resolve_latest_provider_flow_payload(
            source_state=captured
        )
    assert fallback_called is False
