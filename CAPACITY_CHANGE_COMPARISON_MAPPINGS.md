# Capacity Change Comparison: Plant and Train Mapping Logic

**Last Updated:** 2026-04-02

This document explains how the "Capacity Change Comparison in Selected Range" table in `pages/capacity.py` uses country, plant, and train mappings before it builds the comparison rows.

## Scope

The relevant flow is in these functions:

- `load_capacity_source_data()` in `pages/capacity.py`
- `fetch_woodmac_train_capacity_raw_data()` in `pages/capacity.py`
- `fetch_ea_capacity_raw_data()` in `pages/capacity.py`
- `_standardize_country_names()` in `pages/capacity.py`
- `_standardize_plant_names()` in `pages/capacity.py`
- `_apply_train_mapping()` in `pages/capacity.py`
- `_build_train_change_log()` in `pages/capacity.py`
- `_build_ea_change_log()` in `pages/capacity.py`
- `_build_train_change_hierarchical_rows()` in `pages/capacity.py`

The dedicated maintenance pages are:

- `pages/plant_names_mapping.py`
- `pages/train_names_mapping.py`

## High-Level Flow

The comparison table is built in this order:

1. Load the mapping tables from the database.
2. Fetch raw Woodmac and Energy Aspects capacity rows.
3. Standardize `country_name`.
4. Standardize plant names into a shared canonical `plant_name`.
5. Resolve train numbers into a shared canonical numeric `train`.
6. Build provider-specific change logs.
7. Merge Woodmac and Energy Aspects changes into the final comparison table.

The important dependency is:

- Country mapping runs first.
- Plant mapping runs second.
- Train mapping runs third.

Train mapping depends on the canonical plant name already being in place.

## Mapping Tables Used

### 1. Country Mapping

Source table:

- `at_lng.mappings_country`

Used by:

- `_standardize_country_names()`

Behavior:

- Raw country names are uppercased into `raw_country_key`.
- They are mapped to canonical `country_name`.
- If no mapping exists, the original country string is kept.

This matters because plant mapping joins on `country_name`, so plant mappings only work as expected after country normalization.

### 2. Plant Mapping

Source table:

- `at_lng.mapping_plant_name`

Columns loaded by `capacity.py`:

- `country_name`
- `provider`
- `source_field`
- `source_name`
- `scope_hint`
- `component_hint`
- `plant_name`

Canonical purpose:

- Reconcile Woodmac `plant_name` and Energy Aspects `project_name` into one shared plant name.

Used by:

- `_standardize_plant_names()`

Actual join keys used by `capacity.py`:

- `country_name`
- `provider`
- `source_field`
- uppercased `source_name`

Important details:

- `source_name` matching is case-insensitive because both sides are uppercased.
- `country_name`, `provider`, and `source_field` must match exactly after trimming.
- `scope_hint` and `component_hint` are loaded and saved, but they are not used by the mapping join logic in `capacity.py`.
- Duplicate rows are reduced case-insensitively on `source_name`; `capacity.py` keeps the first matching row it reads.

Provider-specific usage:

- Woodmac plant mapping:
  - `provider = 'woodmac'`
  - `source_field = 'plant_name'`
  - source column = raw Woodmac `plant_name`
- Energy Aspects plant mapping:
  - `provider = 'energy_aspects'`
  - `source_field = 'project_name'`
  - source column = EA `project_name`

Fallback behavior:

- If no plant mapping matches, the raw source plant/project name remains as `plant_name`.
- `plant_mapping_applied` is set to `False`.

## How Woodmac Raw Plant and Train Names Are Sourced

The Woodmac train-capacity dataset is assembled from the latest Woodmac plant and train metadata before mappings are applied.

Raw fields come from:

- `woodmac_lng_plant_summary`
- `woodmac_lng_plant_train`
- `woodmac_lng_plant_monthly_capacity_nominal_mta`
- `woodmac_lng_plant_train_annual_output_mta`

In `WOODMAC_TRAIN_CAPACITY_QUERY`, the code builds:

- `country_name`
- `plant_name`
- `lng_train_name_short`
- `id_plant`
- `id_lng_train`
- `capacity_mtpa`

After fetch, `fetch_woodmac_train_capacity_raw_data()` copies the pre-mapping values into:

- `raw_plant_name`
- `raw_train_name`

Those raw columns are preserved so later logic can still audit what the original provider labels were.

## Train Mapping

Source table:

- `at_lng.mapping_plant_train_name`

Columns loaded by `capacity.py`:

- `country_name`
- `plant_name`
- `provider`
- `parent_source_field`
- `parent_source_name`
- `source_field`
- `source_name`
- `scope_hint`
- `component_hint`
- `train`
- `allocation_share`
- `notes`

Canonical purpose:

- Reconcile provider train labels into a plant-scoped numeric `train`.

Used by:

- `_apply_train_mapping()`

Actual join keys used by `capacity.py`:

- `country_name`
- canonical `plant_name`
- `provider`
- `parent_source_field`
- uppercased `parent_source_name`
- `source_field`
- uppercased `source_name`

Important details:

- Train mapping happens after plant mapping.
- The `plant_name` in `mapping_plant_train_name` must match the canonical plant name produced by plant mapping.
- `parent_source_name` and `source_name` are matched case-insensitively.
- `scope_hint`, `component_hint`, and `notes` are not used by the runtime mapping logic.
- Duplicate rows are reduced case-insensitively on `parent_source_name` and `source_name`; `capacity.py` keeps the first matching row it reads.

Provider-specific usage:

- Woodmac train mapping:
  - `provider = 'woodmac'`
  - `parent_source_field = 'plant_name'`
  - `parent_source_column = raw_plant_name`
  - `source_field = 'lng_train_name_short'`
  - `source_column = raw_train_name`
- Energy Aspects train mapping:
  - `provider = 'energy_aspects'`
  - `parent_source_field = 'project_name'`
  - `parent_source_column = project_name`
  - `source_field = 'train_name'`
  - `source_column = train_name`

### Allocation Share

`allocation_share` lets one raw train alias map into one or more canonical train numbers.

Runtime behavior:

- The mapped `train` is assigned.
- `capacity_mtpa` is multiplied by `allocation_share`.

This is why the train mapping page requires:

- a positive numeric `train`
- a positive `allocation_share`
- `allocation_share` sums to `1.0` for each raw source row

That rule allows split mappings such as one raw train label being allocated across multiple canonical trains.

## Direct Train Inference Before Exception Mapping

If there is no explicit row in `mapping_plant_train_name`, `capacity.py` still tries to infer a numeric train from simple names.

The main patterns are:

- exact labels like `Train 1`
- variants like `Train 1 bolt-on`
- parent labels that already contain a train number

This inference is intentionally conservative:

- it only applies to rows that do not already have an explicit mapping
- it only applies when the inferred train is unique for that country/plant context
- it is blocked when an explicit mapping already reserves the same train number for that plant

So the order is:

1. Explicit train mapping from `mapping_plant_train_name`
2. Direct inference from simple raw labels
3. Leave `train` unresolved if neither is safe

The summary text in `capacity.py` describes this as:

- country mapping
- plant mapping
- direct `Train N` inference
- exception mappings from `mapping_plant_train_name`

That wording is directionally correct for the feature, but the implementation detail is slightly more precise:

- explicit mapping is checked first inside `_apply_train_mapping()`
- direct inference is only used for unmapped rows

## How Mappings Affect the Change Log

### Woodmac

`_build_train_change_log()` builds the Woodmac change rows after mapping has already been applied.

Its grouping key depends on whether the train was resolved:

- If `train` is resolved:
  - series key = canonical `country + plant + train`
- If `train` is unresolved:
  - series key = raw fallback key using `raw_plant_name`, `raw_train_name`, `id_plant`, and `id_lng_train`

This is the core reason mapping quality changes the table structure:

- better mapping collapses aliases into shared canonical train rows
- missing mapping leaves fragmented raw series

### Energy Aspects

`_build_ea_change_log()` also runs after plant/train normalization.

For EA:

- `Plant` is the canonical `plant_name` when available, otherwise `project_name`
- `Train` is the canonical numeric `train` when resolved

## Why the Comparison Table Sometimes Falls Back to Plant Level

The final comparison table is built by `_build_train_change_hierarchical_rows()`.

For `Plants + Trains View`, the code only shows train rows when the plant is fully resolved at train level across the visible data.

This is controlled by `_collect_unresolved_train_keys()`:

- it scans the Woodmac and EA period data
- if any row for a `(Country, Plant)` pair has `Train = null`, that plant is marked unresolved

Then `_build_flat_plants_trains_rows()` applies an all-or-nothing rule:

- resolved plant: show train-level rows
- unresolved plant: show one plant-level row instead

Practical consequence:

- one unmapped train alias for a plant can prevent the comparison table from breaking that plant out into shared train rows

This behavior is intentional. It avoids mixing:

- some canonical train rows
- some still-unmapped residual plant-level changes

inside the same visible plant block.

## What the "Unmapped Plant" Section Is Actually Doing

The capacity page includes an unmapped plant review section, but it only covers plant aliases, not train aliases.

Flow:

1. `render_unmapped_plant_table()` rebuilds the Woodmac and EA change logs for the current filters.
2. `_build_unmapped_plant_alias_df()` collects rows where `Mapping Applied = False`.
3. It groups them by:
   - `country_name`
   - `provider`
   - `source_field`
   - `source_name`
4. It suggests `scope_hint` and `component_hint` from the source name when possible.
5. `save_unmapped_plant_mappings()` writes the reviewed rows back into `at_lng.mapping_plant_name`.

Important limitation:

- This section only surfaces plant aliases that appear in the current selected range and that contribute to visible change rows.
- There is no equivalent inline save workflow in `capacity.py` for train mappings.

Train mappings are maintained through the dedicated `Train Names Mapping` page.

## What Is and Is Not Used at Runtime

Used at runtime in `capacity.py`:

- country mapping keys and canonical country name
- plant mapping keys and canonical `plant_name`
- train mapping keys
- canonical numeric `train`
- `allocation_share`

Not used at runtime in `capacity.py`:

- `scope_hint`
- `component_hint`
- `notes`

Those fields are still valuable for curation and documentation, but they do not change the join logic.

## Practical Debugging Checklist

If a plant or train is not appearing correctly in the comparison table, the fastest checks are:

1. Confirm country normalization first.
   - If `country_name` is off, plant mapping will miss.
2. Confirm the plant mapping row.
   - `provider`
   - `source_field`
   - `source_name`
   - canonical `plant_name`
3. Confirm the train mapping row uses the canonical plant name, not the raw alias.
4. Confirm `parent_source_field` and `source_field` are correct for the provider.
5. Confirm `parent_source_name` and `source_name` match the raw provider labels.
6. Confirm `allocation_share` is valid and sums to `1.0` across split mappings.
7. If no explicit train mapping exists, check whether the raw label is simple enough for direct inference.
8. If the plant still stays at plant level in `Plants + Trains View`, look for any unresolved `Train = null` rows for that same `(Country, Plant)` pair.

## Short Summary

The comparison table only becomes truly cross-provider at train level after three layers of normalization succeed:

1. canonical country
2. canonical plant
3. canonical numeric train

Plant mapping is the bridge from raw provider project/plant aliases into one shared plant. Train mapping is the bridge from raw provider train labels into one shared plant-scoped train number. If either layer is missing, the comparison still works, but it becomes less merged and more likely to stay at plant level instead of train level.
