# Scenario Models: Capacity vs Production

This note summarizes how scenarios work today in:

- the `Capacity` page
- the `Production` page / legacy `Terminals`-adjustment flow

The two systems are often discussed as if they were interchangeable. They are not. They use different source data, different persistence tables, and different business logic.

## Purpose

The goal of this note is to make the difference between the two scenario models explicit before any migration or cleanup work is done.

In short:

- `Capacity` scenarios are a **train-event capacity model**
- `Production` scenarios are a **monthly output override model**

That distinction matters because replacing one with the other is not a simple UI change.

## Capacity Page Scenario Model

The Capacity page uses a structured internal scenario model that is edited primarily through the `Train Timeline - Monthly` section.

### External source inputs

The Capacity page reads provider data from:

- `at_lng.woodmac_lng_plant_monthly_capacity_nominal_mta`
- `at_lng.woodmac_lng_plant_train_annual_output_mta`
- `at_lng.woodmac_lng_plant_summary`
- `at_lng.woodmac_lng_plant_train`
- `at_lng.ea_lng_liquefaction_projects`

Before the provider data is used, it is standardized with mapping tables:

- country mapping
- `at_lng.mapping_plant_name`
- `at_lng.mapping_plant_train_name`

### Internal persistence tables

Internal scenarios are stored in:

- `at_lng.fundamentals_capacity_scenarios`
- `at_lng.fundamentals_capacity_scenario_rows`

### One scenario row represents

One saved Capacity scenario row is one train-level event with:

- one train identity
- one event date
- one signed capacity value
- one optional note

The main stored fields are:

- `scenario_row_key`
- `country_name`
- `plant_name`
- `train_label`
- `base_provider`
- `base_first_date`
- `base_capacity_mtpa`
- `scenario_first_date`
- `scenario_capacity_mtpa`
- `scenario_note`

This is intentionally a **snapshot-based event model**, not a free-form monthly curve editor.

### Core logic

The Capacity page logic works as follows:

1. Woodmac and Energy Aspects provider data are loaded into page memory stores.
2. Provider data is converted into signed train-level change logs.
3. Those change logs are converted into timeline rows, with additions and reductions treated separately.
4. When an internal scenario is selected, its saved rows are overlaid onto that same train-level structure.
5. A monthly internal scenario schedule is derived by carrying each row's `scenario_capacity_mtpa` forward from `scenario_first_date` onward.

In other words:

- the DB stores train events
- the monthly country schedule is derived from those events

### Scenario creation paths

The Capacity page supports four scenario creation paths:

- `Woodmac`
- `Energy Aspects`
- `Current Scenario`
- `Existing Internal Scenario`

Provider-based creation is built from provider change history in memory, not from the visible top table alone.

### Save behavior

Capacity scenarios are saved as a **full snapshot replacement**:

- the current scenario rows are normalized
- existing rows for that scenario are deleted
- the full current row set is inserted back into `fundamentals_capacity_scenario_rows`

So the saved state is always the complete current train-event snapshot for that scenario.

## Production Page Scenario / Adjustment Model

The Production page still uses the legacy scenario model inherited from the old `Terminals` flow.

### Baseline source tables

The Production / Terminals logic uses Woodmac output data from:

- `at_lng.woodmac_lng_plant_train_monthly_output_mta`
- `at_lng.woodmac_lng_plant_train_annual_output_mta`

Monthly data takes precedence where available, and annual data is expanded to fill gaps when monthly coverage does not exist.

### Scenario adjustment table

The scenario overlay is stored in:

- `at_lng.fundamentals_terminals_output_adjustments`

### Core logic

The Production page scenario logic works as follows:

- `base_view` = pure Woodmac output baseline
- any other scenario = Woodmac baseline plus `adjusted_output` overrides

The override is applied at:

- plant
- train
- year
- month

using the latest saved adjustment row for that combination.

In practice:

- if an adjustment exists for a given plant/train/month, `adjusted_output` replaces the baseline output
- if no adjustment exists, the Woodmac baseline output remains in place

### Start date logic

In the Production / Terminals model, train start dates are not stored as explicit scenario event rows.

Instead, start dates are inferred from output:

- the effective train start date is the first month with output under the selected scenario
- the `new capacity only` filter depends on those inferred start dates

So this model is fundamentally an **output override model first**, with timing shifts emerging as a consequence of changed monthly output.

### Save behavior

The Production / Terminals scenario system is **append-only**:

- new adjustment rows are inserted into `fundamentals_terminals_output_adjustments`
- the table keeps historical versions
- the latest timestamp wins for any given plant/train/year/month/scenario combination

This is different from the full-snapshot replacement behavior used by the Capacity page.

## Side-by-Side Comparison

| Topic | Capacity page | Production page / legacy adjustments |
| --- | --- | --- |
| Scenario meaning | Internal train-event capacity scenario | Monthly output override scenario |
| Core unit | One train event | One plant/train/month output override |
| External source inputs | Woodmac capacity data, Woodmac metadata, Energy Aspects projects, mapping tables | Woodmac monthly output and annual output |
| Internal persistence | `fundamentals_capacity_scenarios` + `fundamentals_capacity_scenario_rows` | `fundamentals_terminals_output_adjustments` |
| Monthly values | Derived by carrying event capacity forward from `scenario_first_date` | Directly overridden month by month with `adjusted_output` |
| Start dates | Explicit scenario event dates | Inferred from first output month under the scenario |
| Scenario creation model | Structured creation from Woodmac, EA, current scenario, or existing internal scenario | Ad hoc output adjustments on top of Woodmac |
| Save strategy | Full-snapshot replace | Append-only, latest timestamp wins |
| Business model type | Capacity-based | Output-based |

## Practical Implication

Replacing the Production-page scenario with the Capacity-page internal scenario is **not a dropdown swap**.

It is a move from:

- an **output-adjustment model**

to:

- a **capacity-event model**

That means any migration needs to decide whether the Production page should:

- become capacity-led and derive its displayed values from Capacity scenarios
- remain an output page with a separate translation from capacity events into output
- or support both models during a transition

## Two Future Integration Paths

If the Production page is moved onto the Capacity scenario framework, there are two distinct ways to do it.

### 1. Capacity scenario always leads production timing

In this option, the Capacity scenario becomes the hard driver of when capacity is available.

The logic would be:

- the Capacity scenario always determines the effective start date of a new train
- the Capacity scenario also determines the effective date of any additional capacity on an existing train
- Woodmac remains the production reference curve
- Production is then shifted relative to that Woodmac reference when Capacity timing differs

For a **new train**:

- if the Capacity scenario pulls the train forward versus Woodmac, the production profile must move forward
- if the Capacity scenario delays the train versus Woodmac, the production profile must move back

That move should **not** be a flat or naive month-for-month translation. The Woodmac production shape needs to keep its seasonal pattern, so the translated production should preserve the reference seasonality rather than simply moving a constant monthly output block.

For an **existing train with an additional capacity step-up**:

- the extra capacity should be treated as its own timing event
- it should not be merged blindly into the original train history
- the incremental production associated with that additional capacity should move according to the Capacity scenario date, again using Woodmac as the production reference and preserving seasonality

In this model, the Capacity page is the source of truth for **when capacity exists**, and Production becomes a derived view that translates capacity timing into production timing.

### 2. Production can be fully overridden, but capacity start is a hard constraint

In this option, monthly production can still be overridden directly, but the Capacity scenario defines the earliest valid start of supply.

The logic would be:

- production can be fully changed by internal view
- but no production should be allowed before the corresponding Capacity scenario start date
- the start date of new trains and incremental capacity on existing trains must always respect the Capacity scenario

This means the Capacity scenario acts as a hard timing boundary, while production remains flexible after that point.

For a **new train**:

- monthly production can be overridden after the Capacity scenario start
- but the train cannot produce before that start date, regardless of what the production override says

For an **existing train capacity increase**:

- the incremental production tied to the added capacity cannot begin before the Capacity scenario increase date
- after that date, production can still be overridden

In this model, the Capacity page remains the source of truth for **when capacity becomes available**, but Production keeps a larger degree of freedom on **how output evolves after capacity exists**.

### Key difference between the two options

- **Option 1** makes Production a derived output view based on Capacity timing plus Woodmac seasonal reference behavior
- **Option 2** keeps Production as an override model, but constrained so that output can never start before Capacity allows it

The most important shared rule in both options is:

- the Capacity scenario must always control the start timing of new trains and of additional capacity on existing trains
- existing-train capacity increases must be treated as separate timing events, not absorbed into the original train without distinction

## Recommended Future Design

The safest design is a hybrid with a strict hierarchy:

1. **Capacity timing leads**
2. **Woodmac provides the production reference shape**
3. **Manual production override exists only as an explicit exception layer**
4. **Final validation prevents output from violating capacity timing**

This means the default model should follow **Option 1**, while the flexibility of **Option 2** should only exist as a controlled override on top of the capacity-led baseline.

### Why this is the recommended approach

This approach minimizes unwanted overrides because it avoids using month-by-month manual output replacement as the primary scenario engine.

If direct production override remains the main mechanism:

- it becomes too easy to overwrite baseline behavior silently
- it becomes harder to distinguish timing changes from shape changes
- existing-train capacity expansions can accidentally overwrite the original train profile
- reductions and partial capacity changes become difficult to trace

By contrast, a capacity-led design keeps the scenario logic explicit:

- Capacity decides **when** a block of supply exists
- Woodmac reference data decides the default production **shape**
- manual override only changes output when a user intentionally asks for it

### Proposed layering

The future production build should run in layers:

1. **Capacity event layer**
   - use the Capacity scenario as the source of truth for train timing
   - treat each positive capacity row as its own capacity block
   - treat each reduction row as a separate negative event
   - treat existing-train capacity increases as incremental blocks, not as edits to the original train block

2. **Reference production-shape layer**
   - derive a default production profile from Woodmac
   - preserve ramp behavior and seasonality
   - do not use a flat month-for-month shift of raw output

3. **Optional manual override layer**
   - allow explicit production overrides only after the capacity-led baseline is built
   - scope overrides clearly by train or capacity block and time period
   - require notes / justification for traceability

4. **Validation and clipping layer**
   - zero output before the capacity scenario start date
   - prevent incremental production from starting before the related capacity increase date
   - make any clipping or forced adjustment visible to the user

### Handling new trains

For a new train:

- the Capacity scenario start date must always be the effective production start boundary
- Woodmac should be used as the default production reference profile
- if the train is moved earlier or later than Woodmac, the production profile should move with it while preserving the Woodmac seasonal pattern

This should be done through a translated or regenerated reference profile, not through a naive block shift that ignores seasonality.

### Handling additional capacity on existing trains

Additional capacity on an existing train must be handled as a separate capacity block.

That means:

- the original train production remains tied to the original capacity block
- the incremental capacity has its own start date
- the incremental production should be derived separately from the reference production profile
- the two components should only be summed after they are built independently

This is the key protection against unwanted overrides on mature existing-train production.

### Handling reductions

Reductions should not be implemented as a vague netting exercise at country or plant level.

Instead:

- a reduction should remove future availability starting from its event date
- the production effect should be linked to a specific capacity block whenever possible
- if a temporary heuristic is needed, it should be deterministic and visible rather than silent

### Woodmac annual-only data and seasonality

One important design requirement is the treatment of Woodmac annual data from **2029 onward**, where monthly production data is not available for part of the horizon.

In that case:

- the annual Woodmac output should **not** be translated into a flat `1/12` monthly profile for future scenario generation
- a seasonality profile must be applied to convert annual data into monthly reference output
- that seasonal monthly shape should then be used as the production reference when translating capacity timing into production timing

So for annual-only Woodmac periods:

- Woodmac remains the reference source
- but the future monthly production path must be seasonally distributed before any timing translation is applied

### Guardrails to avoid unwanted overrides

To avoid hidden or accidental overrides, the future model should enforce these rules:

- no production before the Capacity scenario says capacity exists
- no implicit replacement of untouched months through broad `COALESCE` logic
- no merging of incremental capacity into the original train without keeping lineage
- no silent reduction allocation
- no final monthly output value without provenance

Every final monthly production point should be explainable through provenance such as:

- capacity-derived baseline
- Woodmac seasonal reference used
- manual override applied or not
- clipped by capacity timing or not
- clipped by another hard rule or not

### Recommended precedence

The final precedence order should be:

1. Capacity scenario event timing
2. Derived active capacity by train / capacity block
3. Woodmac-based production reference profile with seasonality
4. Explicit manual production override
5. Final validation and clipping checks

This is the cleanest way to keep the Capacity scenario in control while still preserving flexibility where needed.

## References

Current implementation:

- `dash_shipping_lng_snd/pages/capacity.py`
- `dash_shipping_lng_snd/pages/production.py`
- `dash_shipping_lng_snd/pages/terminals.py`
- `fundamentals/terminals/capacity_scenario_utils.py`
- `fundamentals/terminals/scenario_utils.py`

Current documentation:

- `dash_shipping_lng_snd/pages/capacity_train_timeline_monthly.md`
- `dash_shipping_lng_snd/pages/TERMINALS_DATA_SOURCES.md`
