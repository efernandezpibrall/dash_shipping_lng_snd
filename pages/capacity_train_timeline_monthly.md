# Train Timeline - Monthly

This document explains how the `Train Timeline - Monthly` section on the Capacity page works today.

The implementation lives mainly in:

- `dash_shipping_lng_snd/pages/capacity.py`
- `fundamentals/terminals/capacity_scenario_utils.py`

## 1. What the section is for

`Train Timeline - Monthly` is a combined reference-and-editing view.

It does two jobs at the same time:

1. It shows a side-by-side timeline summary for the two external providers:
   - Woodmac
   - Energy Aspects
2. When an internal scenario is selected, it becomes the scenario editor for that same train-level structure.

So conceptually:

- without an internal scenario selected, it behaves like a reference timeline table
- with an internal scenario selected, it becomes an editable planning surface

## 2. High-level flow

The section is built in four layers:

1. Source data is loaded into page memory stores
2. Woodmac and Energy Aspects are converted into train-level timeline summaries
3. If an internal scenario is selected, its rows are overlaid onto that summary
4. The result is rendered in AG Grid and can be exported to Excel

The main functions involved are:

- `_build_train_timeline_df`
- `_build_train_timeline_grid_rows`
- `_create_train_timeline_table`
- `render_train_timeline_table`
- `export_train_timeline_excel`

## 3. Source data feeding the section

The section depends on the same page-level source stores used elsewhere on the Capacity page.

The raw provider inputs are loaded in `load_capacity_source_data` and stored in:

- `capacity-page-train-capacity-data-store`
- `capacity-page-ea-capacity-data-store`

Those raw inputs are then used in two different ways:

- selected countries
- `Other Countries` mode

For row inclusion:

- the visible Train Timeline rows still come from what falls inside the current `Date Range`

For cross-source lookup:

- the section can also read matching provider / scenario rows outside the selected `Date Range`
- but only to backfill missing source cells for rows that are already visible

So the Train Timeline is still anchored to the current page selection, but it is no longer limited to showing only in-range values in every source column.

## 4. How Woodmac and Energy Aspects are converted into the timeline

### Woodmac / Energy Aspects visible timeline logic

The provider rows are built from change logs, not from the top capacity tables.

For the visible `Train Timeline - Monthly` section, the pipeline is:

1. Build a provider change log inside the current page selection
2. Keep signed change rows, not just positive additions
3. Within that visible window, collapse each canonical `Country / Plant / Train` into separate:
   - reduction rows
   - addition rows
4. For each sign, show:
   - the first visible date for that sign
   - the total signed capacity change for that sign inside the visible window
5. After the visible row set is built, backfill missing source cells from the nearest same-sign row outside the selected range when needed

This logic is implemented through:

- `_build_train_change_log` for Woodmac
- `_build_ea_change_log` for Energy Aspects
- `_build_provider_timeline_snapshot`
- `_build_train_timeline_df`

### What “First Date” means here

For Woodmac and Energy Aspects inside `Train Timeline - Monthly`, `First Date` has two possible meanings.

If the source has an in-range visible row, it is:

- the first visible date for that signed row
- within the current visible window
- after country filtering and `Other Countries` handling have been applied

If the source is being backfilled from outside the range, it is:

- the nearest matching same-sign row outside the selected range
- preferring the row before the range when both sides are equally close

So in practice:

- a reduction row uses the first visible reduction month
- an addition row uses the first visible addition month
- an out-of-range backfill still has to use the same sign as the visible row

### What “Capacity” means here

For Woodmac and Energy Aspects inside this section, `Capacity` means:

- the signed capacity change represented by that visible row
- within the current visible window

For aggregated provider rows, that is the total visible capacity change for one sign:

- total visible reductions for the reduction row
- total visible additions for the addition row

If a source cell is backfilled from outside the selected range, the value shown is the exact nearest same-sign out-of-range row for that source rather than a netted opposite-sign substitute.

That is why the note below the grid now describes provider rows as signed visible changes with same-sign out-of-range backfills when needed.

## 5. How the provider timeline table is merged

`_build_train_timeline_df` still merges the provider snapshots on:

- `timeline_reference_key`
- `Country`
- `Plant`
- `Train`

Then `_build_train_timeline_grid_rows` applies sign-aware row matching for the visible grid and fallback layer using:

- `Country`
- `Plant`
- `Train`
- sign / direction

That sign-aware layer is what prevents an addition row from ever being filled by a reduction row, or vice versa.

The merged table contains provider-side columns such as:

- `Woodmac First Date`
- `Woodmac Capacity Change`
- `Energy Aspects First Date`
- `Energy Aspects Capacity Change`

If the `Original names` toggle is enabled, the table also shows:

- `Woodmac Original Name`
- `Energy Aspects Original Plant`
- `Energy Aspects Original Train`

These original-name fields are built from the raw provider alias fields that survived mapping and aggregation. For Woodmac, the Train Timeline uses the source-backed full `lng_train_name` label rather than showing separate raw plant and raw train columns.

## 6. Internal scenario model

When an internal scenario is selected, the Train Timeline becomes an editor for scenario rows.

The internal scenario model is snapshot-based and DB-backed.

The two persistence tables are:

- `at_lng.fundamentals_capacity_scenarios`
- `at_lng.fundamentals_capacity_scenario_rows`

They are managed through `fundamentals/terminals/capacity_scenario_utils.py`.

### One scenario row represents

Each scenario row stores:

- a stable `scenario_row_key`
- `country_name`
- `plant_name`
- `train_label`
- `base_provider`
- `base_first_date`
- `base_capacity_mtpa`
- `scenario_first_date`
- `scenario_capacity_mtpa`
- `scenario_note`
- display sort fields

This means the editable scenario model is intentionally simple:

- one date
- one capacity
- one optional saved note
- one row per train event

It is not a free-form monthly curve editor.

## 7. How new scenarios are created

The `Create New Scenario` card supports four creation paths:

- `Woodmac`
- `Energy Aspects`
- `Current Scenario`
- `Existing Internal Scenario`

The creation logic is handled in `manage_capacity_scenario_state`.

### If created from Woodmac

The app builds scenario rows from the Woodmac change log using:

- `_build_new_capacity_scenario_rows(..., "woodmac", ...)`
- `_build_provider_scenario_rows_from_change_log`

Important detail:

- Woodmac scenarios are created from signed change events
- not from the top capacity matrix
- not from the visible Train Timeline row set

In other words:

- the created baseline comes from the full provider change history available in memory
- current country filters do not limit which source rows are used to create the scenario

The selected page `Date Range` still matters for provider-based scenario creation, but only as a split point:

- rows strictly before the selected start month are kept as exact signed event rows
- rows from the selected start month onward are collapsed into separate future reduction and future addition rows per canonical train

That design allows duplicated source scenarios to preserve reductions as well as additions.

### If created from Energy Aspects

The same idea applies, but using the Energy Aspects change log.

That also means:

- it is created from the full provider change history available in memory
- not only from what is currently visible in the page selection
- and the selected `Date Range` start month becomes the split point between exact history and aggregated future rows

### If created from Current Scenario

The app uses the active working copy in memory, which means unsaved edits can be branched into a new scenario.

### If created from Existing Internal Scenario

The app copies the saved scenario rows of the selected existing internal scenario.

## 8. How the selected scenario is loaded

Scenario selection is driven by the sticky `Internal Scenario` dropdown at the top of the page.

When the dropdown changes, `handle_capacity_scenario_selection`:

1. loads the selected scenario rows from the DB
2. may rebuild legacy rows if needed
3. stores the working copy in memory
4. updates the selected-scenario state

The `Selected Scenario` card inside Train Timeline is just showing the active internal scenario state that came from this shared selector.

One important branch:

- if the current scenario has unsaved user edits, switching scenarios is not immediate
- the app first asks for confirmation before discarding the working copy

## 9. Legacy rebuild behavior

There is a legacy repair path for old Woodmac / Energy Aspects scenarios:

- `_maybe_rebuild_legacy_capacity_scenario_rows`

This exists because earlier internal scenarios were created with a less complete event model.

If the app detects a scenario that still matches its original base row-for-row and qualifies as legacy, it can rebuild a corrected working copy from the current provider baseline.

Important current behavior:

- this rebuild can happen on load
- but it no longer marks the scenario as `Unsaved edits`
- `Unsaved edits` is now reserved for user-originated edits

## 10. What the AG Grid shows

The Train Timeline grid is created by:

- `_build_train_timeline_grid_rows`
- `_get_train_timeline_grid_column_defs`
- `_create_train_timeline_table`

### Always-visible reference fields

The grid uses the merged provider timeline as its reference backbone when provider rows exist, and shows:

- `Country`
- `Plant`
- `Train`
- Woodmac first date and capacity change
- Energy Aspects first date and capacity change

If visible scenario rows exist that do not have a matching provider reference row in the current selection, those scenario rows can still appear in the grid as scenario-only rows.

### When `Original names` is set to `Show`

The grid prepends the provider original-name fields. For Woodmac this is a single `Original Name` column sourced from raw Woodmac `lng_train_name`; for Energy Aspects the grid still shows separate `Original Plant` and `Original Train` columns.

### When an internal scenario is selected

The grid adds editable internal scenario columns:

- `Scenario First Date`
- `Scenario Capacity`
- `Scenario Note`

The editable user-input columns are:

- `Scenario First Date`
- `Scenario Capacity`
- `Scenario Note`

## 11. What can be edited

When a scenario is selected:

- `Scenario First Date` is editable
- `Scenario Capacity` is editable
- `Scenario Note` is editable

The grid uses AG Grid editors:

- date-string editor for dates
- numeric editor for capacity
- popup text editor for notes

Practical note:

- a note can be saved even when a row does not introduce a new scenario date or capacity override

The grid also enables:

- undo / redo
- keyboard selection / text copy
- row highlighting

## 12. How row highlighting works

Rows are visually highlighted when the internal scenario differs from its base values.

The flag is:

- `__scenario_overridden`

That is calculated in `_build_train_timeline_grid_rows`.

A row is considered overridden when either:

- scenario first date differs from base first date
- scenario capacity differs from base capacity

## 13. What counts as “Unsaved edits”

This is an important detail.

`Unsaved edits` now means only user-originated changes, not system refreshes.

The chip is driven by:

- `_build_capacity_scenario_dirty_label`
- `mark_capacity_scenario_dirty`
- `add_capacity_scenario_row`

It becomes `Unsaved edits` only when:

- you edit `Scenario First Date` in the grid
- you edit `Scenario Capacity` in the grid
- you edit `Scenario Note` in the grid
- you add a new row through `Add Row to Selected Scenario`

It does not turn on merely because:

- the page refreshed
- a scenario was selected
- a legacy scenario was auto-rebuilt into a corrected working copy

## 14. How grid edits are resolved

The page does not immediately write every cell edit to the database.

Instead:

1. AG Grid emits row data / cell events
2. the page keeps a working copy in memory
3. `Save` explicitly persists that working copy

That working copy now includes optional row notes as part of the saved internal scenario snapshot.

The key functions are:

- `_update_working_scenario_rows_from_grid`
- `_resolve_active_capacity_scenario_rows`
- `mark_capacity_scenario_dirty`

### Why this matters

This design makes editing faster because every keystroke does not trigger a DB write.

## 15. Add Row behavior

The `Add Row to Selected Scenario` card allows manually adding a new scenario row.

The required inputs are:

- `Country`
- `Plant`
- `First Date`
- `Capacity`

`Train` is optional.

The logic lives in:

- `_append_manual_capacity_scenario_row`
- `add_capacity_scenario_row`

Validation rules:

- country is required
- plant is required
- train can be blank, but if provided it must be a positive whole number
- first date must parse as a valid date and is normalized to the month start
- capacity must be greater than `0`

In practice, the UI prompts for `YYYY-MM`, which is the intended input format.

The new row is added to the working copy first, not directly to the DB.

## 16. Save / Revert / Delete

These actions apply to the currently selected internal scenario.

### Save

`Save`:

- resolves the current working rows from the grid
- writes them to the DB via `save_capacity_scenario_rows`
- clears the dirty state

### Revert

`Revert`:

- reloads the last saved scenario rows from the DB
- discards the current working copy
- clears the dirty state

### Delete

`Delete`:

- deletes the whole selected internal scenario
- not just a row
- requires confirmation first

### Upload via Excel

The selected scenario card also supports round-tripping Train Timeline through Excel.

Workflow:

1. export the Train Timeline workbook
2. edit it offline
3. upload it back into the selected scenario
4. review the staged result in the grid
5. click `Save`

Important behavior:

- upload is available only when an internal scenario is selected
- upload is blocked when the current working copy already has unsaved edits
- only `.xlsx` files exported from this section are accepted
- a valid upload updates the scenario working copy and marks it as `Unsaved edits`
- upload does not write directly to the DB; `Save` is still required

Validation rules:

- `Train` can be blank, but if provided it must be a positive whole number
- existing exported rows may only change:
  - `Scenario First Date`
  - `Scenario Capacity`
  - `Scenario Note`
- changing `Country`, `Plant`, `Train`, provider columns, or original-name columns causes the upload to be rejected
- new rows must provide:
  - `Country`
  - `Plant`
  - `Scenario First Date`
  - `Scenario Capacity`
- new rows may leave `Train` blank, but if they populate it then it must be a positive whole number
- new rows must leave provider/original-name columns blank
- deleting a provider-only reference row is not allowed
- deleting an exported row that maps to a real internal-scenario row is treated as a row delete request

The delete confirmation now explicitly warns that the scenario will be deleted and the action cannot be reverted.

## 17. How Date Range affects the section

The selected page `Date Range` affects Train Timeline in multiple ways.

### For Woodmac and Energy Aspects

It defines the visible provider window used by the Train Timeline section.

That means:

- provider rows use signed capacity changes visible in the current selection
- reductions and additions are shown independently
- each provider row uses the first visible date for that sign and the total visible signed change for that sign
- rows are still created only from in-range activity
- missing source cells can now be backfilled from the nearest same-sign row outside the selected range
- opposite-sign rows are never used as fallbacks

For provider-based scenario creation, the same selected start month is also used as the split point between:

- exact historical signed rows before that month
- aggregated future reduction / addition rows from that month onward

### For internal scenarios

The visible scenario rows are filtered by:

- `scenario_first_date` if present
- otherwise `base_first_date`

This filtering is handled in `_filter_visible_capacity_scenario_rows`.

So yes: the Date Range applies to the internal scenario side of Train Timeline as well.

But the selected internal scenario can also provide out-of-range fallback values for an already-visible `Country / Plant / Train / sign` row when there is no in-range scenario row for that sign.

## 18. How country filters affect the section

The section uses the same shared country filtering logic as the rest of the Capacity page.

The visible row set is based on:

- selected countries
- `Other Countries` include/exclude mode

The available countries themselves are built from:

- Woodmac monthly scope
- Energy Aspects monthly scope
- internal scenario monthly schedule

This is why the Train Timeline and the rest of the page stay aligned when filters change.

## 19. How the internal scenario ties into other sections

The Train Timeline editor is not isolated.

Its working copy also feeds:

- `Internal Scenario Capacity`
- `Capacity Change Comparison in Selected Range`
- Excel exports

That means if you edit the scenario timeline and then save it, those other views can use the same scenario state.

## 20. How the internal scenario monthly schedule is derived

The top `Internal Scenario Capacity` section does not read the grid directly.

Instead, it derives a monthly schedule from the scenario rows via:

- `_build_internal_scenario_monthly_schedule`

Current model:

- each scenario row becomes active from `scenario_first_date`
- that signed capacity value is carried forward month by month to the selected end date

So the current model is best thought of as a single-step carried-forward event model:

- positive rows act like additions from their effective month onward
- negative rows act like reductions from their effective month onward

It is not yet a multi-step retirement / ramp-up engine.

## 21. How the comparison table uses the scenario

The comparison table uses:

- `_build_internal_scenario_change_log`
- `_prepare_internal_period_change_df`

This now exposes the scenario as a single comparison column:

- Net Delta

The same selected internal scenario is therefore used consistently in:

- Train Timeline
- Internal Scenario Capacity
- Capacity Change Comparison in Selected Range

## 22. How the bottom LNG Train Timeline chart uses the section

The Capacity page also includes a second `LNG Train Timeline` chart at the bottom.

That chart does not query a separate legacy scenario pipeline.

Instead, it reads the current visible `Train Timeline - Monthly` row data directly, so it stays aligned with:

- the current country filter
- the current `Date Range`
- same-sign row matching
- out-of-range backfill flags
- current unsaved internal scenario edits in the grid

The chart adds two controls:

- `Source`
- `Compare To`

Available choices are:

- `Woodmac`
- `Energy Aspects`
- the currently selected internal scenario, when one is selected

Behavior:

- bars are drawn from the selected `Source` only
- additions stay upward and reductions stay downward
- arrows appear only when `Source` and `Compare To` are different and both dates exist on the same visible row identity
- if a visible source row has no same-sign `Compare To` date at all, the bar is marked instead of drawing an arrow:
  - amber outline when the full plant / month / sign bucket is source-only
  - amber badge when only part of the aggregated bucket is missing in `Compare To`
- those arrows therefore reuse the same visible row logic as the Train Timeline grid instead of recomputing a separate comparison universe

## 23. Export behavior

`export_train_timeline_excel` exports the same timeline logic used by the visible section.

It:

1. rebuilds the provider timeline for the current selection
2. resolves the active internal scenario state, including current grid edits when needed
3. applies the same same-sign out-of-range backfill logic used by the visible grid
4. exports the merged grid-shaped result

It also respects:

- selected countries
- `Other Countries` mode
- `Date Range`
- `Original names` toggle
- whether an internal scenario is selected

When an internal scenario is selected, the export also includes the saved `Scenario Note` column.

For round-trip upload support, that scenario export also carries:

- a hidden row key column on the main `Train Timeline` sheet
- a hidden metadata sheet used to validate re-imports against the selected scenario and original exported rows

So the export is intended to mirror the visible Train Timeline state closely.

In the grid itself, highlighted source cells indicate that the value was backfilled from outside the selected `Date Range`.

## 24. Important nuance: provider rows vs scenario rows

Provider rows and scenario rows do not mean the same thing.

Provider-side timeline rows are hybrid reference rows:

- signed visible provider changes inside the current Train Timeline window
- with reductions and additions kept separate
- with visible matching treated at `Country / Plant / Train / sign` level
- and with missing source cells optionally backfilled from the nearest same-sign row outside the selected range
- and, for provider-based scenario creation, exact pre-start history preserved before the selected start month

Internal scenario rows are event rows:

- one scenario date
- one scenario capacity
- one optional saved note
- can be positive or negative

The note is descriptive only:

- it does not change schedule math
- it does not change row matching
- it does not affect charts or comparison totals

That difference is intentional:

- provider rows are reference baselines derived from source change logs
- internal scenario rows are editable planning inputs

## 25. Practical mental model

The easiest way to think about the section is:

- rows appear because at least one source has signed activity in the selected range
- Woodmac and Energy Aspects columns first show what that selected range currently implies
- then blank source cells can be filled from the nearest same-sign row outside the range
- provider-based scenario creation preserves full signed source history before the selected start month and simplifies the forward window into separate reduction and addition rows
- the internal scenario columns let you define your own alternative event row set
- the internal scenario note column lets you attach optional saved context to a row
- saving the scenario persists that custom row set
- the top internal scenario chart/table and the comparison section then consume it

## 26. Summary

`Train Timeline - Monthly` is the operational center of the internal scenario workflow.

It:

- shows provider-side signed change rows inside the current page selection, with reductions and additions treated independently
- keeps visible row matching sign-aware at `Country / Plant / Train / sign`
- backfills missing source cells from the nearest same-sign out-of-range row when available
- lets you create or select an internal scenario
- lets you edit scenario first dates and capacities
- lets you save an optional note per internal scenario row
- lets you add manual scenario rows
- feeds the internal scenario chart, comparison table, bottom LNG train timeline chart, and exports
- keeps edits in memory until you explicitly save

In short:

- provider columns are the baseline reference
- internal scenario columns are the editable plan
- the selected internal scenario is the shared state that ties the page together
