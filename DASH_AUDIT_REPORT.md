# Dash Shipping LNG SND — Technical Audit and Optimization Report

Audit dates: 2026-08-19 to 2026-08-20
Scope: application-owned Dash, Python, SQL, delivery, cache, and test code in this repository. Imported write/data helpers in the private sibling `fundamentals` package were treated as an explicit audit boundary.

## Executive result

The application keeps the same valid-input business outputs and visible page behavior, while closing correctness, security, multi-worker, delivery, and maintainability gaps.

- All 19 registered routes render, with 202 server callbacks and 3 clientside callbacks.
- The original 609 cases pass; the final suite has 636 unique cases after adding 27 regression cases for the repaired failure modes.
- Test modules were consolidated from 32 to 6 (`-81.25%`) without dropping an original test node.
- Production Python is 89,442 physical lines versus 89,631 in the frozen starting workspace (`-189` net). The work removed or consolidated 806 lines of dead, unused, or exact-duplicate production code; the smaller net reduction reflects new cross-worker snapshot, validation, delivery, and lock-safety code.
- Valid-output parity was checked independently after every implementation batch using callback/layout fingerprints, exact serialized Dash trees, DataFrame schema/value hashes, figure/export hashes, security probes, focused tests, and full suites.
- No live database schema, index, role, or production data mutation was performed during the audit.

The baseline was the complete workspace state frozen at the start of the audit, including its pre-existing uncommitted files, rather than only Git `HEAD`.

## Implemented changes

### 1. Safe code reduction

- Removed 612 unreachable lines after unconditional returns in `pages/terminals.py` and `pages/terminal_adjustments.py`.
- Removed 5 physical lines for three genuinely unused local assignments.
- Preserved 31 apparent `F401` imports after regression tests proved they are compatibility exports used by callers/tests.
- Moved the exact duplicate 178-line Supply/Demand comparison-card builder and identical style constant to `utils/balance_components.py`, reducing production by another 189 physical lines while retaining byte-identical component trees.

Gross removal/consolidation: 806 production lines. Net production change after all reliability code: `-189` lines.

### 2. Faster startup and static delivery

- Deferred heavy callback/export-only imports such as Plotly Express, openpyxl, requests, and dash-leaflet.
- Enabled compression for streamed Dash assets and immutable one-year caching for versioned asset URLs.
- Preserved exact decoded asset bytes with compression enabled and disabled.
- Retained orjson as an explicit runtime dependency so Plotly can select its faster JSON engine.

Measured in six alternating fresh-process pairs:

| Metric | Before | After | Change |
|---|---:|---:|---:|
| Cold app import, median | 789.901 ms | 733.118 ms | `-7.19%` |
| Max RSS, median | 234,225,664 B | 210,436,096 B | `-10.16%` |
| macOS peak footprint, median | 189,899,436 B | 169,075,360 B | `-10.97%` |

The audited static payload was 1,269,035 bytes. Its gzip-9 size was 386,453 bytes, a potential cold-transfer reduction of 882,582 bytes (`69.5%`) when the client accepts gzip.

### 3. Terminal-adjustment SQL injection closure

- Treats browser-provided plant/train selections as untrusted.
- Resolves them against the authoritative database-backed allowlist.
- Rejects malformed, unknown, and duplicate selections before any baseline read or write.
- Uses SQLAlchemy bind parameters rather than interpolating plant/train strings into SQL.

The hostile value `O'Brien' OR 1=1 --` was verified absent from the SQL text; it remained only a bound value. Valid train-copy results, frames, columns, and scenarios remained exact.

### 4. Explicit snapshot failure behavior

- Importer period refresh now propagates `SnapshotUnavailable` with a recovery instruction instead of returning a valid-looking empty dataset.
- Contracts consumers now return visible recovery content for missing, corrupt, evicted, or malformed references rather than producing callback 500s or silently using an unrelated process-global snapshot.

Ordinary empty-data and normal error behavior remains unchanged. The only intended differences are for corrupt/untrusted inputs that were previously unsafe or misleading.

### 5. Contracts multi-worker correctness and reuse

The deployment uses four Gunicorn workers, but Contracts previously kept the full snapshot only in one process and sent the browser a token that other workers could not resolve.

The replacement:

- publishes an immutable shared snapshot reference;
- stores only that small reference in the browser;
- resolves the exact shared Contracts, demand, assumptions, and formula frames in every worker;
- rechecks source identity after construction and never publishes a mixed/drifting build;
- preserves the exact prior browser reference as last-good fallback after refresh failure;
- pairs status text with the snapshot that produced it, removing the status/data race;
- verifies exact disk records without decoding all four DataFrames on the warm availability path;
- republishes reference-less or evicted same-revision process state safely.

An external four-process cold-start proof observed exactly one call to each of the four source loaders (`[1, 1, 1, 1]` total), and a fifth fresh process resolved identical frames, dtypes, indexes, order, nulls, and year settings. Relative to four workers independently loading all four sources, cold full-source reads fall from 16 to 4 (`-75%`).

### 6. Snapshot publication lock correctness

Test consolidation exposed a nondeterministic self-deadlock in the prior 64-stripe file-lock design. The final design:

- retains 64 stripes for concurrent ordinary snapshot builds;
- makes same-thread same-stripe acquisition reentrant without weakening cross-process `flock` exclusion;
- serializes compound multi-artifact publications through one same-host compound lock;
- enforces compound-before-stripe ordering and fails fast on reverse/different-stripe nesting;
- shares the in-process coordinator across overlapping cache-store generations for the same root;
- tracks active descriptors from both current and retired stores;
- resets inherited locks, Futures, ContextVars, cache handles, and active lock descriptors after `fork`;
- rejects stale-PID and closed-store use.

Deterministic regressions cover same-stripe recursion, thread and process ABBA order, reverse hierarchy, current-store fork, retired-store fork, and same-root close/reopen generation overlap. A 30-generation stress run quiesced cleanly.

### 7. Country-map network work bounded and reproducible

- Pinned the Natural Earth GeoJSON to a specific upstream commit.
- Verified the pinned and then-current upstream bytes were identical: 838,726 bytes, SHA-256 prefix `6866c877…`.
- Added a single-flight, per-worker cache so repeated map interactions make one successful request rather than one per callback.
- Returns a deep copy so callback mutation cannot contaminate the cached source.
- Preserves the prior visible failure component when the fetch fails.

For `N` map renders in one worker, remote requests fall from `N` to `1`, eliminating `N-1` requests and `(N-1) × 838,726` response bytes. Across `W` hosts/workers the cold-load upper bound remains `W`; vendoring the pinned file is the next step if zero cold network dependency is required.

### 8. Schema configurability

Removed active executable `at_lng.` literals from Contracts, Shipping Balance, exporter/importer detail SQL, and the shared snapshot table. All now use `DB_SCHEMA`.

- Under the current `at_lng` configuration, affected SQL text, bind metadata, and representative results were byte-identical.
- In an isolated non-default-schema process, all 18 audited SQL objects/constants resolved to `alternate_lng`, with no executable `at_lng.` literal.

### 9. Reproducible runtime manifest

Replaced the partial performance-only manifest with `requirements.txt`, containing 17 exact pins that cover every public top-level import plus the non-lexical runtime dependencies used by Dash compression, Gunicorn, SQLAlchemy's psycopg2 dialect, and Plotly's orjson selection.

The private sibling `fundamentals` package remains an external deployment dependency and must be present on `PYTHONPATH`. The audited workstation also has an unrelated installed `trino` distribution that requests `requests>=2.32.4`, while this app's active/pinned Requests version is 2.31.0; `trino` is not imported by this repository, but the deployment environment should be isolated or the dependency set reconciled before a fresh build.

### 10. Test-suite consolidation

The 32 test modules contained no exact duplicate test bodies, so no semantic case was deleted. They were reorganized into the minimum practical domain boundaries:

1. `tests/test_app_and_operations.py`
2. `tests/test_dashboard_snapshot_cache.py`
3. `tests/test_exporters.py`
4. `tests/test_fleet_metrics_source_refs.py`
5. `tests/test_importers.py`
6. `tests/test_shared_market_and_physical.py`

All 629 pre-consolidation nodes map exactly into the new files, including parametrization suffixes, with no duplicate nodes. Seven lock/fork regressions bring the final total to 636. Both default discovery order and the mapped historical order are exercised because the consolidation itself revealed an order-sensitive lock defect.

Test lines were not removed at the expense of evidence: physical test source grew from 20,828 frozen-baseline lines to 22,066 because the repaired behaviors gained 27 regression cases and the merged files retain source-provenance separators.

## Output-preservation evidence

- 202 server callbacks and 3 clientside callbacks remain registered.
- Final callback fingerprint: SHA-256 prefix `71116786…`; its intentional change from baseline is the hidden Contracts shared-reference/status wiring.
- All 19 normalized route layouts match; no duplicate component IDs were introduced.
- Supply/Demand's four extracted cards and both full layouts serialize byte-for-byte identically, including ordered IDs, text, dropdown options/defaults, clearable flags, classes, styles, and `n_clicks`.
- Contracts normal figures, tables, filters, workbook exports, and four resolved source frames match the frozen baseline.
- Representative Shipping, mapping, Market Balance, LNG Physical, Capacity, exporter, and importer figures/exports match exact semantic hashes.
- Static asset decoded SHA values match with compression on and off.
- Browser QA covers all routes at desktop width, aliases/redirects, `/health`, `/ready`, asset headers, console errors, visible content, and Dash error overlays.
- A 1440×1200 Exporters capture is pixel-identical to the frozen baseline (zero differing channels; SHA-256 `05f5b705c8fef59aa4cddf6ba73a47e789068d88c03db6b571f38305698bfe48`).
- Independent final runs passed 636/636 three times in default order (42.57 s, 40.89 s, and 40.80 s) and once in mapped historical order (41.14 s), with no deadlock, timeout, warning, or drift.

The test suite's runtime depends strongly on process/cache state. Final full runs are approximately 41–43 seconds; the original first frozen run was about 79 seconds. This report does not attribute that difference to test-file consolidation.

## Remaining prioritized findings

These items need deployment ownership, database change control, business policy, or a larger behavior decision, so they were audited but not changed speculatively.

### P1 — database plans and indexes

Read-only live planner inspection found:

- a WoodMac flow path with four sequential scans over roughly 3.2 million rows and estimated cost around 539k;
- a Platts current-forecast path with three sequential scans over roughly 2.17 million rows (about 1.17 GB) and estimated cost around 361k;
- route/distance work scanning and hashing an unindexed 109,462-row distance matrix;
- full-table `MAX(...)` watermark scans on large source tables.

Candidate work, to validate with `EXPLAIN (ANALYZE, BUFFERS)` in staging:

- composite indexes matching dataset/snapshot or market-outlook/vintage/date predicates;
- indexes supporting the latest upload/publication timestamp lookups;
- an `(origin_node_name, destination_node_name)` distance-pair index;
- precomputed/materialized latest-vintage slices where repeated scans dominate.

Do not create these blindly: index write/storage cost, actual selectivity, ingestion patterns, and concurrent build behavior must be measured first. The distance matrix also has 127 duplicated name pairs with differing distances. Current latest delivered trades did not hit those pairs, but the data should be repaired and a uniqueness/tie-break policy established before enforcing a unique index.

### P1 — deployment security and privileges

- No application-level authentication/authorization layer is visible in this repository, including around write-capable callbacks. Confirm that SSO/authorization is enforced by the reverse proxy or platform; otherwise add role checks before deployment.
- The audited database role had `SELECT`, `INSERT`, `UPDATE`, and `DELETE` on raw `kpler_trades` and administrative tables. Split read-only dashboard access from narrowly scoped editor credentials and execute write callbacks under the least-privileged role.
- Upload callbacks base64-decode CSV/XLSX content without an application business-size limit. Configure Flask/request limits and validate decoded workbook size, sheet count, row count, and decompression ratio before parsing.

### P1/P2 — worker/process topology

- Four workers with the configured defaults `pool_size=5` and `max_overflow=5` can permit up to 40 simultaneous SQLAlchemy connections. Set `DASH_DB_POOL_SIZE` and `DASH_DB_MAX_OVERFLOW` from an explicit database connection budget; a `3 + 2` profile would cap four workers at 20, but it must be load-tested against real concurrency.
- Capacity refresh jobs and executor state remain process-local. A worker recycle can lose job state, and overlapping callbacks can race shared `running` outputs. Move long-running jobs to a production queue/shared state backend such as Celery/Redis if those workflows must survive worker replacement.
- The immutable snapshot disk cache is shared across workers on one host. It is not a multi-replica coordination system. Use a shared backend and distributed lock if the application is deployed to multiple hosts/containers.

### P2 — frontend/accessibility maintenance

- `assets/styles.css` is 225,719 bytes and contains 490 `!important` declarations. Refactor by page/token in a visual-change project rather than during preserve-output work.
- The static audit found 26 hidden native choice inputs, a pre-existing H1→H3 heading skip on `/exporters`, and small chart modebar/grid filter hit targets. These should be addressed with keyboard/screen-reader testing.
- Inspected Plotly wrappers and AG Grid `treegrid` regions lack `aria-label`/`aria-labelledby`; nearby visible headings do not fully replace programmatic region names.
- Mobile layouts avoid page-level horizontal overflow down to 320 px, but navigation relies on a local horizontal scroller and later routes begin off-screen.
- Capacity renders transparently with EA and WoodMac upload timestamps shown as “Unavailable”; repair the upstream freshness metadata rather than hiding the missing identity.
- Market Balance contains repeated internal Plotly SVG IDs such as `symbol`; these are library-generated internals rather than duplicate Dash component IDs, but should be rechecked after Plotly upgrades.

### P2 — source identity and data quality

- A `COUNT/MAX(timestamp)` watermark cannot detect an in-place correction that preserves both values. Contracts deliberately rebuilds on an explicit global Refresh for this reason. Prefer immutable ingestion revision IDs/checksums where available.
- The pinned map payload is still fetched once per cold worker. Vendor the exact pinned file or serve it from an internal immutable object store if external availability is unacceptable.

## Deployment checklist

1. Put the private `fundamentals` package and the exact `requirements.txt` environment on `PYTHONPATH`/the runtime image.
2. Configure an explicit per-worker database connection budget and confirm total connections against the database limit.
3. Confirm SSO/authorization and split read-only/editor database roles.
4. Add request and decoded-upload limits before enabling uploads to untrusted users.
5. Provision an owner-only persistent snapshot directory; use a shared backend instead for multiple hosts.
6. Test candidate indexes in staging with representative `EXPLAIN (ANALYZE, BUFFERS)` and ingestion load.
7. Run `PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests -q`, compile checks, route/browser smoke, and callback/layout fingerprints in CI.
8. Monitor structured snapshot events for build, decode, fallback, corruption, publication, and lock-wait latency.

## Primary guidance used

- Dash performance: <https://dash.plotly.com/performance>
- Dash data sharing across callbacks/workers: <https://dash.plotly.com/sharing-data-between-callbacks>
- Dash background callbacks and production queue guidance: <https://dash.plotly.com/background-callbacks>
- Dash background callback caching: <https://dash.plotly.com/background-callback-caching>
- SQLAlchemy connection pooling: <https://docs.sqlalchemy.org/en/20/core/pooling.html>
- Flask request-size configuration: <https://flask.palletsprojects.com/en/stable/config/#MAX_CONTENT_LENGTH>
- PostgreSQL indexes: <https://www.postgresql.org/docs/17/indexes.html>
- PostgreSQL examining index usage: <https://www.postgresql.org/docs/17/indexes-examine.html>
- PostgreSQL transaction isolation: <https://www.postgresql.org/docs/17/transaction-iso.html>
