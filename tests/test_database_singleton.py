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

    assert pool.size() == database.DEFAULT_POOL_SIZE == 5
    assert pool._max_overflow == database.DEFAULT_MAX_OVERFLOW == 5
    assert pool._timeout == database.DEFAULT_POOL_TIMEOUT_SECONDS == 30
    assert pool._recycle == database.DEFAULT_POOL_RECYCLE_SECONDS == 1800
