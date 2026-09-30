# LNG Shipping Dashboard

## Runtime and sources

- [index_shipping_snd.py](index_shipping_snd.py) owns the page registry, aliases,
  navigation, and live page imports. [Procfile](Procfile) serves
  `index_shipping_snd:server`; `app.py` alone does not register the full application.
- [utils/database.py](utils/database.py) reads `DASHBOARD_CONFIG_FILE`, falling back
  to `../config.ini`. Page imports can access sources; inspect the affected import
  path before using an import as a smoke check.
- For snapshot changes, start with [utils/dashboard_snapshot_cache.py](utils/dashboard_snapshot_cache.py).
  Backend selection and fallback behavior depend on runtime flags; same-host
  persistence does not establish support for multiple application hosts.

## Current contracts

Preserve these unless the requested change explicitly changes the contract:

- Physical demand annualization requires twelve distinct months with numeric
  values; incomplete years stay missing. Global supply comparisons include only
  complete periods, with April–September summer and October–March winter.
- Snapshot references identify an exact source revision. Renderers and exports
  must resolve that revision; preserve each caller's unavailable/fallback behavior.
- Routing changes must account for aliases and navigation in the page registry,
  not only the imported page module.

## Verification

From this checkout, use the project interpreter (`../.venv/bin/python` is available
in this workspace). Full suite: `python -m pytest tests -q`; select affected files
for local changes. Period/data contracts are in `tests/test_shared_market_and_physical.py`;
snapshot coverage is in `tests/test_dashboard_snapshot_cache.py`, with fleet
consumer checks in `tests/test_fleet_metrics_source_refs.py` and route/operations
checks in `tests/test_app_and_operations.py`.

`tests/conftest.py` uses `setdefault` to enable revision/fleet rollout flags. Check
the relevant environment when comparing test and deployment behavior; these flags
do not prove source isolation or coverage of disabled paths.
