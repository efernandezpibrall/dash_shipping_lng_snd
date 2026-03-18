# Global LNG Shipping Balance: Technical Architecture & Methodology

**Document Version:** 2.1
**Last Updated:** 2025-09-30
**Purpose:** Comprehensive technical documentation for top-tier shipping analysts

---

## ⚠️ CRITICAL METHODOLOGICAL FIXES

### Version 2.1 Updates (2025-09-30)

**1. Mean Vessel Capacity Calculation Fix**

**Issue Identified:** Aggregations were using `mean_vessel_capacity=('vessel_capacity_cubic_meters', 'mean')`, which calculates the **mean of means** at multiple aggregation levels. This is mathematically incorrect when aggregating pre-aggregated data because it gives equal weight to each group regardless of the number of vessels/trades.

**Mathematical Problem:**
```
Group A: 10 trades using 100,000 m³ vessels → mean = 100,000 m³
Group B:  5 trades using 200,000 m³ vessels → mean = 200,000 m³

WRONG: mean(100,000, 200,000) = 150,000 m³  (equal weight to each group)
RIGHT: (10×100,000 + 5×200,000) / (10+5) = 133,333 m³  (weighted by trade count)
```

**The Fix:** Calculate `mean_vessel_capacity_cubic_meters = sum_vessel_capacity / count_trades` at every aggregation level instead of taking mean of means. This ensures proper weighted averaging across vessel types, regions, and time periods.

**Locations Fixed:**
- `kpler_analysis()`: Lines 261, 333 - Intracountry and shipping region aggregations
- `calculate_shipping_metrics()`: Lines 685, 731 - Vessel type aggregations for future metrics
- `calculate_historical_metrics()`: Lines 846, 858 - Vessel type aggregations for historical metrics

**2. Regional Breakdown UI Enhancement**

Added vessel type multi-dropdown filters to Regional Breakdown tables in both Demand and Supply views (`shipping_balance.py`). Currently displays available vessel types from Kpler data but requires backend modifications for full filtering functionality (vessel_type dimension is aggregated away in current implementation).

---

## ⚠️ CRITICAL METHODOLOGICAL FIX (Version 2.0 - 2025-09-30)

### **Vessel Capacity Alignment: The Apples-to-Apples Fix**

**Issue Identified:** Original implementation used **trade-pattern vessel capacity** to calculate ships_demand, then compared against **fleet vessel count**. This created fundamental distortion because:

- **Demand calculation** used `mean_vessel_capacity` from trade patterns (varies by route, region, vessel type mix)
- **Supply calculation** counted fleet vessels uniformly (each vessel = 1 ship, regardless of size)
- **Result:** Artificial inflation/deflation of net balance depending on whether trades used smaller/larger vessels than fleet average

**Real-World Impact Example:**
```
Fleet Composition: 400 vessels, average 160,000 cbm
Trade Patterns:    Predominantly 145,000 cbm vessels on certain routes

OLD METHOD:
ships_demand = volume / (145,000 cbm × utilization) × round_trip_time = 337.8 vessels
net = 400 - 337.8 = 62.2 vessels surplus

CORRECTED METHOD:
ships_demand = volume / (160,000 cbm × utilization) × round_trip_time = 306.2 vessels
net = 400 - 306.2 = 93.8 vessels surplus

Distortion: 31.6 vessels (52% overstatement of market tightness)
```

**The Fix:** Replace trade-based `mean_vessel_capacity` with fleet-based `average_size_cubic_meters` in ships_demand calculation. This ensures:
- **Demand side:** How many vessels *from the actual fleet* are needed
- **Supply side:** How many vessels *the actual fleet* has
- **Net balance:** True market tightness based on consistent vessel capacity assumption

**Location of Fix:** `fundamentals/shipping_balance_calculator.py`, lines 1461-1474

---

## Executive Summary

This document provides an exhaustive technical breakdown of the Global LNG Shipping Balance calculation system, covering both **Demand View** (import-centric) and **Supply View** (export-centric) perspectives. The system integrates vessel fleet analysis, historical trade patterns, and forward-looking capacity forecasts to produce actionable shipping balance metrics at multiple temporal granularities.

**Key Technical Achievements:**
- Dual-perspective analysis (demand-side import terminal vs. supply-side export terminal viewpoints)
- Volume-weighted aggregation methodology eliminating statistical bias
- Time-accurate denominator calculation preventing artificial volatility
- Seamless historical-to-forecast transition with configurable utilization assumptions
- Multi-temporal granularity support (monthly, quarterly, seasonal, yearly)
- **Vessel capacity alignment ensuring apples-to-apples net balance comparison (v2.0)**

---

## 1. System Architecture Overview

### 1.1 Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA INGESTION LAYER                         │
├─────────────────────────────────────────────────────────────────────┤
│  1. Kpler Trades (at_lng.kpler_trades)                              │
│  2. WoodMac Forecasts (at_lng.woodmac_gas_imports_exports_monthly)  │
│  3. Vessel Fleet Data (at_lng.kpler_vessels_info, at_lng.syy_newbuilds) │
│  4. Mapping Tables (at_lng.mappings_country, at_lng.mapping_vessel_type_capacity) │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│                   CORE CALCULATION ENGINE                            │
│          (shipping_balance_calculator.py)                            │
├─────────────────────────────────────────────────────────────────────┤
│  Step 1: Fetch WoodMac Import/Export Forecasts (fetch_woodmac_data) │
│  Step 2: Determine Historical Cutoff (get_max_historical_date)      │
│  Step 3: Calculate Future Shipping Metrics (calculate_shipping_metrics) │
│  Step 4: Calculate Historical Metrics (calculate_historical_metrics) │
│  Step 5: Process Shipping Demand (process_shipping_demand)          │
│  Step 6: Analyze Vessel Fleet Lifecycle (analyze_vessel_fleet)      │
│  Step 7: Calculate Final Balance (calculate_final_balance)          │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      VISUALIZATION LAYER                             │
│            (dash_shipping_lng_snd/pages/shipping_balance.py)        │
├─────────────────────────────────────────────────────────────────────┤
│  - Demand View Chart & Table                                        │
│  - Supply View Chart & Table                                        │
│  - Regional Trade Analysis                                          │
│  - Intracountry Trade Breakdown                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Module Responsibilities

| Module | Primary Responsibility | Key Functions |
|--------|----------------------|---------------|
| `shipping_balance_calculator.py` | Core shipping balance logic | `global_shipping_balance()`, `kpler_analysis()`, `calculate_shipping_metrics()` |
| `shipping_balance.py` (page) | Dash application interface | UI callbacks, chart generation, user interaction |
| `kpler_fundamentals.py` | Historical trade pattern extraction | (Referenced but calculation done in shipping_balance_calculator) |

---

## 2. The Heart of the System: `shipping_balance_calculator.py`

### 2.1 Master Function: `global_shipping_balance()`

**Location:** `fundamentals/shipping_balance_calculator.py:371-436`

**Function Signature:**
```python
def global_shipping_balance(
    engine: Engine,
    aggregation_level: str = 'monthly',
    life_expectancy: int = 20,
    lng_view: str = 'demand',
    utilization_rate: float = 0.85,
    window_end_date: Optional[Union[str, pd.Timestamp]] = None
) -> pd.DataFrame
```

**Parameters Deep Dive:**

1. **`engine`** (SQLAlchemy Engine)
   - Database connection object
   - Used for all SQL queries to PostgreSQL database
   - Assumes connection to schema `at_lng`

2. **`aggregation_level`** (str, default: 'monthly')
   - Controls temporal granularity of output
   - Options: `'monthly'`, `'quarterly'`, `'seasonal'`, `'yearly'`
   - **Critical Impact:** Affects days-in-period calculation, directly influencing ships_demand denominator
   - Seasonal definition: Winter (Oct-Mar), Summer (Apr-Sep)

3. **`life_expectancy`** (int, default: 20)
   - Expected operational lifespan of LNG vessels in years
   - Used to calculate theoretical retirement dates
   - Distribution algorithm spreads retirements across 12 months to avoid clustering

4. **`lng_view`** (str, default: 'demand')
   - **'demand'**: Import-centric perspective
     - Laden vessels measured at **destination** (discharge port)
     - Non-laden vessels measured at **origin** (where they departed for ballast leg)
   - **'supply'**: Export-centric perspective
     - Laden vessels measured at **origin** (loading port)
     - Non-laden vessels measured at **destination** (where they arrive for next cargo)
   - **Why This Matters:** Determines which end of the voyage is used for metric aggregation

5. **`utilization_rate`** (float, default: 0.85)
   - Target vessel capacity utilization for **future forecasts only**
   - Range: 0.0 to 1.0 (typically 0.75-0.95)
   - Historical periods use **realized utilization rates** from actual trade data
   - Higher values = fewer ships needed (assumes more efficient operations)

6. **`window_end_date`** (Optional date)
   - End date of 365-day rolling window for future projections
   - Window: [window_end_date - 365 days, window_end_date]
   - If None: defaults to recent data (2024+)
   - **Purpose:** Allows analyst to choose which historical patterns inform future projections
   - Example: Setting to 2024-12-31 uses 2024 patterns; setting to 2023-06-30 uses mid-2022 to mid-2023 patterns

---

## 3. The Seven-Step Calculation Process

### Step 1: Fetch WoodMac Forecasts (`fetch_woodmac_data`)

**Location:** Lines 475-624

**Purpose:** Retrieve import/export volume forecasts that drive shipping demand

**Technical Details:**

**SQL Query Architecture:**
```sql
WITH latest_short_term AS (
    -- Fetch latest Short Term Outlook data
    -- Aggregated at specified temporal level
    -- Converts MMTPA to cubic meters (m³): metric_value / time_divisor * 2222 * 1000
),
short_term_max_date AS (
    -- Find cutoff between short-term and long-term forecasts
),
latest_long_term AS (
    -- Fetch Long Term Outlook for dates beyond short-term coverage
),
combined_data AS (
    -- Union short-term and long-term forecasts
),
enriched_data AS (
    -- Add temporal dimensions: year, quarter, season, month
    -- Define season: W (Oct-Mar), S (Apr-Sep)
)
SELECT * FROM enriched_data
LEFT JOIN at_lng.mappings_country -- Add shipping region classification
```

**Aggregation Logic by Level:**

| Level | SQL Grouping | Time Divisor | Output Frequency |
|-------|-------------|--------------|-----------------|
| Monthly | `country_name, start_date::DATE` | 12 | One row per country-month |
| Quarterly | `country_name, DATE_TRUNC('quarter', start_date)` | 4 | One row per country-quarter |
| Seasonal | Custom CASE statement | 2 | One row per country-season |
| Yearly | `country_name, DATE_TRUNC('year', start_date)` | 1 | One row per country-year |

**Critical Transformation:**
- Original data in MMTPA (Million Metric Tons Per Annum)
- Conversion: `metric_value / time_divisor * 2222 * 1000 = cubic meters (m³) for LNG cargo`
- Time divisor adjusts annual rate to period rate

**Output DataFrame Structure:**
```
Columns: ['Country', 'Date', 'value', 'season', 'quarter', 'year', 'month', 'source', 'shipping_region']
- 'value' in cubic meters (m³) for the period
- 'Date' is period start date (behavior varies by aggregation level):
  • Monthly: Actual month start (e.g., 2024-01-01, 2024-02-01)
  • Quarterly: First day of quarter (e.g., 2024-01-01 for Q1, 2024-04-01 for Q2)
  • Seasonal: Jan 1 for Winter (W), Jul 1 for Summer (S) (e.g., 2024-01-01, 2024-07-01)
  • Yearly: First day of year (e.g., 2024-01-01)
- 'season', 'quarter', 'year', 'month' columns are ALWAYS populated regardless of aggregation level
  • These are extracted from the Date column for reference and downstream filtering
  • Example: Monthly aggregation for 2024-02-15 → season='W', quarter='Q1', year=2024, month=2
- 'shipping_region' from mapping table (e.g., 'US Gulf', 'North Europe')
```

---

### Step 2: Determine Historical Cutoff (`get_max_historical_date`)

**Location:** Lines 627-638

**Purpose:** Establish clean boundary between realized history and forecast

**SQL Query:**
```sql
SELECT max("end") as hist_date_max
FROM at_lng.kpler_trades
WHERE status='Delivered'
AND upload_timestamp_utc = (SELECT max(upload_timestamp_utc) FROM at_lng.kpler_trades)
```

**Logic:**
1. Find latest delivered trade in most recent Kpler data upload
2. Extract the month of that trade
3. Return **last day of the PREVIOUS complete month** (using `pd.offsets.MonthEnd(-1)`)

**Why Previous Month-End:** Ensures only complete months of data are used in historical calculations, preventing partial-month artifacts

**Example:**
- Latest delivered trade: 2024-09-15
- `hist_date_max` returns: 2024-08-31 (end of previous complete month)
- All dates ≤ 2024-08-31 = historical (use realized metrics)
- All dates > 2024-08-31 = forecast (use projected metrics)
- September 2024 is excluded because it's incomplete at time of data snapshot

---

### Step 3: Calculate Future Shipping Metrics (`calculate_shipping_metrics`)

**Location:** Lines 641-776

**Purpose:** Generate shipping pattern projections for future forecast periods

**Window Mechanism:**

```python
if window_end_date:
    window_end = pd.to_datetime(window_end_date)
    window_start = window_end - pd.Timedelta(days=365)  # 365-day window
else:
    window_start = pd.Timestamp('2024-01-01')  # Default to recent patterns
    window_end = None
```

**Critical Function Call:**
```python
_, df_trades_shipping_region = kpler_analysis(
    engine,
    start_date=window_start,
    end_date=window_end,
    aggregation_level=aggregation_level
)
```
- Fetches and pre-aggregates trades within the window
- Returns **volume-weighted metrics** at target aggregation level
- All subsequent operations use these pre-aggregated values

**View-Specific Column Selection Logic:**

**Important Note:** Data from `kpler_analysis()` is **already aggregated** at the `[shipping_region + time_cols]` level. The code simply selects the relevant columns and renames them - no further aggregation is needed.

**For Supply View:**
```python
# Laden voyages: measure at ORIGIN (export terminal)
# Select pre-aggregated metrics at origin
select_cols_laden = ['origin_shipping_region'] + time_cols + ['mean_delivery_days', 'sum_cargo_origin', 'count_trades']
mean_shipping_days_laden = recent_df_laden[select_cols_laden].copy()
mean_shipping_days_laden.rename(columns={
    'origin_shipping_region': 'shipping_region',
    'mean_delivery_days': 'mean_laden_days'
}, inplace=True)

# Non-laden voyages: measure at DESTINATION (where vessel arrives for loading)
select_cols_nonladen = ['destination_shipping_region'] + time_cols + ['mean_delivery_days', 'count_trades']
mean_shipping_days_nonladen = recent_df_nonladen[select_cols_nonladen].copy()
mean_shipping_days_nonladen.rename(columns={
    'destination_shipping_region': 'shipping_region',
    'mean_delivery_days': 'mean_nonladen_days'
}, inplace=True)

# Cargo metrics: from laden trades at origin
select_cols_cargo = ['origin_shipping_region'] + time_cols + [
    'mean_cargo_origin_cubic_meters',
    'aggregate_utilization_rate_origin',
    'mean_vessel_capacity_cubic_meters'
]
median_cargo_volumes = recent_df_laden[select_cols_cargo].copy()
median_cargo_volumes.rename(columns={
    'origin_shipping_region': 'shipping_region',
    'mean_cargo_origin_cubic_meters': 'mean_cargo_cubic_meters',
    'aggregate_utilization_rate_origin': 'aggregate_utilization_rate'
}, inplace=True)
```

**For Demand View:**
```python
# Laden voyages: measure at DESTINATION (import terminal)
# Select pre-aggregated metrics at destination
select_cols_laden = ['destination_shipping_region'] + time_cols + ['mean_delivery_days', 'sum_cargo_destination', 'count_trades']
mean_shipping_days_laden = recent_df_laden[select_cols_laden].copy()
mean_shipping_days_laden.rename(columns={
    'destination_shipping_region': 'shipping_region',
    'mean_delivery_days': 'mean_laden_days'
}, inplace=True)

# Non-laden voyages: measure at ORIGIN (where vessel departs for ballast leg)
select_cols_nonladen = ['origin_shipping_region'] + time_cols + ['mean_delivery_days', 'count_trades']
mean_shipping_days_nonladen = recent_df_nonladen[select_cols_nonladen].copy()
mean_shipping_days_nonladen.rename(columns={
    'origin_shipping_region': 'shipping_region',
    'mean_delivery_days': 'mean_nonladen_days'
}, inplace=True)

# Cargo metrics: from laden trades at destination
select_cols_cargo = ['destination_shipping_region'] + time_cols + [
    'mean_cargo_destination_cubic_meters',
    'aggregate_utilization_rate_destination',
    'mean_vessel_capacity_cubic_meters'
]
median_cargo_volumes = recent_df_laden[select_cols_cargo].copy()
median_cargo_volumes.rename(columns={
    'destination_shipping_region': 'shipping_region',
    'mean_cargo_destination_cubic_meters': 'mean_cargo_cubic_meters',
    'aggregate_utilization_rate_destination': 'aggregate_utilization_rate'
}, inplace=True)
```

**Return Dictionary:**
```python
{
    'mean_shipping_days_laden': DataFrame,      # Laden leg transit times by region-period
    'mean_shipping_days_nonladen': DataFrame,   # Ballast leg transit times by region-period
    'median_cargo_volumes': DataFrame           # Cargo size, utilization, vessel capacity by region-period
}
```

**Why "mean" in column names when data is volume-weighted:**
The term "mean" reflects that these values are averages calculated using volume-weighting in the upstream `kpler_analysis()` function. The naming preserves backward compatibility.

**Why no aggregation is needed:**
Since `kpler_analysis()` already returns data aggregated at the `[shipping_region, year, month/quarter/season]` level, each combination of grouping keys represents exactly one row. The code simply selects relevant columns and renames them for downstream merging with WoodMac forecasts.

---

### Step 4: Calculate Historical Metrics (`calculate_historical_metrics`)

**Location:** Lines 810-909

**Purpose:** Extract realized shipping patterns from historical trade data

**Key Difference from Step 3:**
- Uses **all historical data** up to `hist_date_max` (not a rolling window)
- Captures actual realized patterns, not projected ones
- Uses same aggregation logic as Step 3 but with full history

**Function Call:**
```python
_, df_trades_shipping_region = kpler_analysis(
    engine,
    end_date=hist_date_max,
    aggregation_level=aggregation_level
)
```
- No `start_date` specified = uses all available historical data
- `end_date=hist_date_max` ensures no forecast contamination

**View Configuration Lookup:**

| View | Laden Region | Non-Laden Region | Cargo Column | Utilization Column |
|------|--------------|------------------|--------------|-------------------|
| Demand | destination_shipping_region | origin_shipping_region | cargo_destination | utilization_rate_destination |
| Supply | origin_shipping_region | destination_shipping_region | cargo_origin | utilization_rate_origin |

**Output Structure (List of 3 DataFrames):**
1. **Laden Metrics:** `[shipping_region, time_cols, mean_laden_days]`
2. **Non-Laden Metrics:** `[shipping_region, time_cols, mean_nonladen_days]`
3. **Cargo Metrics:** `[shipping_region, time_cols, mean_cargo_cubic_meters, aggregate_utilization_rate, mean_vessel_capacity]`

**Time Columns by Aggregation Level:**

| Level | Time Columns Included |
|-------|----------------------|
| Monthly | `['year', 'month']` |
| Quarterly | `['year', 'quarter']` |
| Seasonal | `['year', 'season']` |
| Yearly | `['year']` |

**Important Note:** Data from `kpler_analysis()` includes `vessel_type` dimension. In `calculate_historical_metrics`, we aggregate across `vessel_type` to produce one row per `[shipping_region, time]` combination, using volume-weighted aggregation:
- Sum cargo volumes and vessel capacities across vessel types
- Recalculate utilization rates and mean cargo sizes from aggregated sums
- Use simple mean for delivery days (already volume-weighted within each vessel_type)

---

### Step 5: Process Shipping Demand (`process_shipping_demand`)

**Location:** Lines 912-1069

**Purpose:** Combine WoodMac forecasts with shipping metrics to calculate vessel demand

**High-Level Flow:**

```python
def process_shipping_demand(...):
    # Split WoodMac data at historical cutoff
    wm_net_imports_mcm_hist = wm_net_imports_mcm[wm_net_imports_mcm.Date <= hist_date_max]
    wm_net_imports_mcm_fut = wm_net_imports_mcm[wm_net_imports_mcm.Date > hist_date_max]

    # Process each separately
    df_shipping_demand_hist = process_historical_demand(wm_hist, historical_metrics, ...)
    df_shipping_demand_fut = process_future_demand(wm_fut, shipping_metrics, ...)

    # Concatenate
    return pd.concat([df_shipping_demand_hist, df_shipping_demand_fut])
```

#### 5.1 Historical Demand Processing (`process_historical_demand`)

**Location:** Lines 945-989

**Merge Strategy with Duplicate Prevention:**
```python
# Determine correct merge keys based on aggregation level
merge_keys = ['shipping_region']
if aggregation_level == 'monthly':
    merge_keys.extend(['year', 'month'])
elif aggregation_level == 'quarterly':
    merge_keys.extend(['year', 'quarter'])  # Need BOTH year and quarter (Q1 2023 ≠ Q1 2024)
elif aggregation_level == 'seasonal':
    merge_keys.extend(['year', 'season'])   # Need BOTH year and season (Summer 2023 ≠ Summer 2024)
elif aggregation_level == 'yearly':
    merge_keys.append('year')

# Merge with comprehensive duplicate checks
for i, metric_df in enumerate(historical_metrics):
    # 1. Check for duplicate column names in metric_df
    # 2. Check for duplicate rows in metric_df based on merge keys
    # 3. Verify row count doesn't increase after merge

    common_cols = [k for k in merge_keys if k in metric_df.columns]
    df = merge(df, metric_df, on=common_cols, how='left')
```

**Key Insight:** Quarter and season labels (like 'Q1', 'Summer') repeat across years, so we **must** include 'year' in the merge keys to uniquely identify time periods. For example, Q1 2023 and Q1 2024 both have `quarter='Q1'` but different `year` values.

**Duplicate Detection System:**
The code includes comprehensive checks at every merge point:

1. **Before Merge - Column Check:**
   ```python
   if len(metric_df.columns) != len(set(metric_df.columns)):
       raise ValueError("Duplicate column names detected")
   ```

2. **Before Merge - Row Check:**
   ```python
   if metric_df.duplicated(subset=merge_keys).any():
       raise ValueError("Duplicate rows in metric data")
   ```

3. **After Merge - Count Check:**
   ```python
   rows_before = len(df)
   df = merge(df, metric_df, on=merge_keys, how='left')
   if len(df) > rows_before:
       raise ValueError("Merge created duplicates")
   ```

These checks ensure data integrity and catch issues immediately with clear error messages.

**Aggregation to Regional Level:**
```python
# Drop rows with missing essential data
df = df.dropna(subset=['mean_laden_days', 'mean_nonladen_days', 'mean_cargo_cubic_meters',
                        'aggregate_utilization_rate', 'mean_vessel_capacity'])

# Aggregate by Date AND shipping_region using VOLUME-WEIGHTED AVERAGING
# 'value' = LNG volume in cubic meters (m³) for each country-period from WoodMac forecasts
wm_value = lambda x: np.round(
    np.average(x, weights=df.loc[x.index, "value"]), 1
)

result = df.groupby(['Date', 'shipping_region']).agg({
    'value': 'sum',  # Total LNG volume (m³) for the region-period across countries in that region
    'mean_cargo_cubic_meters': wm_value,  # Volume-weighted average cargo size within region
    'mean_laden_days': wm_value,          # Volume-weighted average laden transit time within region
    'mean_nonladen_days': wm_value,       # Volume-weighted average ballast transit time within region
    'aggregate_utilization_rate': wm_value,  # Volume-weighted average utilization rate within region
    'mean_vessel_capacity': wm_value      # Volume-weighted average vessel capacity within region
})
```

**What is "value"?**
- `value` = LNG trade volume in **cubic meters (m³)** for each country-region-period
- Comes from WoodMac forecasts: `SUM(metric_value) / time_divisor * 2222 * 1000`
- Used as the **weight** in volume-weighted averaging
- Example: If US imports 1,000,000 m³ and France imports 500,000 m³, US metrics get 2x the weight

**Why Volume-Weighted Averaging:**
- Larger volume regions have more influence on global average
- Prevents distortion from low-volume outlier regions
- Reflects actual shipping demand proportionally
- Example: A region moving 10 million m³ should have 10x the weight of a region moving 1 million m³

#### 5.2 Future Demand Processing (`process_future_demand`)

**Location:** Lines 992-1069

**Merge Strategy:**
Same structure as historical, but merges with future metrics from Step 3

**Merge Keys by Aggregation Level:**
```python
if aggregation_level == 'monthly':
    merge_keys = ['shipping_region', 'year', 'month']
elif aggregation_level == 'quarterly':
    merge_keys = ['shipping_region', 'quarter']
elif aggregation_level == 'seasonal':
    merge_keys = ['shipping_region', 'season']
elif aggregation_level == 'yearly':
    merge_keys = ['shipping_region', 'year']
```

**Missing Data Handling (Supply View Only):**
```python
if missing_data.any() and lng_view == 'supply':
    # Use fallback values from window metrics or existing data
    for col in critical_cols:
        if col in shipping_metrics['median_cargo_volumes'].columns:
            fallback_value = shipping_metrics['median_cargo_volumes'][col].mean()
        elif col in shipping_metrics['mean_shipping_days_laden'].columns:
            fallback_value = shipping_metrics['mean_shipping_days_laden'][col].mean()
        elif col in shipping_metrics['mean_shipping_days_nonladen'].columns:
            fallback_value = shipping_metrics['mean_shipping_days_nonladen'][col].mean()
        else:
            # Use mean from existing data in the dataframe
            fallback_value = df[col].mean()

        df[col] = df[col].fillna(fallback_value)
```
- Supply view less prone to missing data (fewer concentrated export terminals with higher volumes)
- Demand view may have sparse data for smaller import regions or emerging markets
- All fallback values derived from actual data (either window metrics or dataframe means)

**Final Aggregation:**
Same regional volume-weighted aggregation as historical processing (by Date and shipping_region)

---

### Step 6: Analyze Vessel Fleet (`analyze_vessel_fleet`)

**Location:** Lines 1072-1193

**Purpose:** Track global LNG vessel fleet evolution over time

**Data Sources:**

1. **Kpler Vessels Info** (`at_lng.kpler_vessels_info`)
   - Active vessels: operational fleet
   - Inactive vessels with port call history: recently retired
   - Excludes floating storage units (FSUs)

2. **SYY Newbuilds** (`at_lng.syy_newbuilds`)
   - Vessels under construction
   - Firm delivery dates for future fleet additions

**SQL Queries:**

```sql
-- Existing Fleet
SELECT DISTINCT
    "Name" as vessel_name,
    vessel_status,
    vessel_build_year,
    vessel_capacity_cubic_meters,
    last_port_call_end
FROM at_lng.kpler_vessels_info
WHERE (vessel_status ='Active' OR (vessel_status ='Inactive' AND last_port_call_end IS NOT NULL))
    AND vessel_build_year IS NOT NULL
    AND is_floating_storage=FALSE
    AND upload_timestamp_utc = (SELECT max(upload_timestamp_utc) FROM at_lng.kpler_vessels_info)

-- Under Construction
SELECT DISTINCT
    "Name" as vessel_name,
    'In construction' as vessel_status,
    extract('year' from "Delivery") as vessel_build_year,
    "Delivery" as vessel_delivery_date,
    "Capacity" as vessel_capacity_cubic_meters
FROM at_lng.syy_newbuilds
WHERE upload_timestamp_utc = (SELECT max(upload_timestamp_utc) FROM at_lng.syy_newbuilds)
```

**Retirement Date Calculation:**

```python
# Step 1: Calculate theoretical retirement year
vessels_df['theorical_retirement_date'] = pd.to_datetime(
    vessels_df.vessel_build_year + life_expectancy, format='%Y'
)

# Step 2: Prevent backdating (vessels don't retire before Feb 2025)
vessels_df.loc[
    vessels_df.theorical_retirement_date < dt.datetime(2025, 2, 1),
    'theorical_retirement_date'
] = dt.datetime(2025, 2, 1)

# Step 3: Distribute retirements across 12 months (prevent clustering)
# See distribute_retirement_dates() function
```

**Retirement Distribution Algorithm:**

**Problem:** If 50 vessels all built in 2005 with 20-year life expectancy, they'd all retire in Jan 2025, creating artificial spike

**Solution:**
```python
def distribute_retirement_dates(vessels_df):
    grouped = vessels_df.groupby('theorical_retirement_date')
    for retirement_date, group in grouped:
        n = len(group)
        months = np.linspace(0, 11, n, dtype=int)  # Spread across 12 months
        group['theorical_retirement_date'] = [
            retirement_date + pd.DateOffset(months=int(m)) for m in months
        ]
    return combined_groups
```
- `np.linspace(0, 11, n)` creates evenly spaced month offsets
- 10 vessels → retire in months [0, 1.2, 2.4, 3.7, 4.9, 6.1, 7.3, 8.5, 9.8, 11]
- Converts integer → DateOffset

**Final Retirement Date Logic:**

```python
vessels_df['retirement_date'] = np.where(
    vessels_df['vessel_status'] != 'Inactive',
    vessels_df[['upload_timestamp_utc', 'theorical_retirement_date']].max(axis=1),
    vessels_df['last_port_call_end']
)
```
- **Active/Construction vessels:** Retire at later of (today, theoretical retirement date)
  - Ensures active vessels don't appear to have retired in the past
- **Inactive vessels:** Use actual last port call date (already retired)

**Monthly Statistics Calculation:**

```python
def calculate_monthly_vessel_stats(vessels_df):
    results = []
    for date_i in pd.date_range(start='1/1/2000', end='1/1/2040', freq='MS'):
        # Active at start of month
        ships_before_month = vessels_df[
            (vessels_df['vessel_delivery_date'] < date_i) &
            (vessels_df['retirement_date'] > date_i)
        ]

        # Delivered during month
        ships_during_month = vessels_df[vessels_df['vessel_delivery_date'] == date_i]

        # Retired during month
        ships_removed_during_month = vessels_df[vessels_df['retirement_date'] == date_i]

        total_active_ships = len(ships_before_month) + len(ships_during_month)

        avg_ship_size = (
            ships_before_month['vessel_capacity_cubic_meters'].sum() +
            ships_during_month['vessel_capacity_cubic_meters'].sum()
        ) / total_active_ships if total_active_ships > 0 else 0

        results.append({
            'date': date_i,
            'total_active_ships': total_active_ships,
            'ships_added': len(ships_during_month),
            'ships_removed': len(ships_removed_during_month),
            'average_size_cubic_meters': avg_ship_size
        })

    return pd.DataFrame(results)
```

**Output DataFrame:**
```
Columns: ['date', 'total_active_ships', 'ships_added', 'ships_removed', 'average_size_cubic_meters']
One row per month from 2000-2040
```

---

### Step 7: Calculate Final Balance (`calculate_final_balance`)

**Location:** Lines 1194-1276

**Purpose:** Calculate ships demand by region, then aggregate to global for fleet comparison

**Major Architecture Change:**
- Ships demand now calculated at **shipping_region level**
- Regional results provide granular demand by region
- Global results sum regional demands for fleet comparison

**The Core Calculation:**

#### 7.1 Time Denominator Calculation

**Function:** `get_days_in_period()` (Lines 439-472)

**Critical Importance:** Ships demand calculation uses this as denominator. Incorrect values cause artificial volatility.

```python
def get_days_in_period(date_series, aggregation_level):
    if aggregation_level == 'monthly':
        return date_series.dt.days_in_month  # 28-31 days

    elif aggregation_level == 'quarterly':
        quarter_start = date_series.dt.to_period('Q').dt.start_time
        quarter_end = date_series.dt.to_period('Q').dt.end_time
        return (quarter_end - quarter_start).dt.days + 1  # 90-92 days

    elif aggregation_level == 'seasonal':
        # Winter: Oct-Mar = 182 days, Summer: Apr-Sep = 183 days
        month = date_series.dt.month
        return month.map(lambda m: 182 if m in [10,11,12,1,2,3] else 183)

    elif aggregation_level == 'yearly':
        year = date_series.dt.year
        return year.map(lambda y: 366 if (y%4==0 and (y%100!=0 or y%400==0)) else 365)
```

**Example:**
- Q1 2024 (Jan-Mar): 91 days (leap year)
- Q2 2024 (Apr-Jun): 91 days
- Q3 2024 (Jul-Sep): 92 days
- Q4 2024 (Oct-Dec): 92 days

#### 7.2 Ships Demand Formula (By Region)

**The Fundamental Equation (Applied per shipping_region):**

```
                            LNG Volume Demand (Regional)
ships_demand = ────────────────────────────────────────────── × (laden_days + nonladen_days) / days_in_period
               vessel_capacity × utilization_rate
```

**Key Change:** This formula is now calculated **separately for each shipping_region**, then summed for global total.

**CRITICAL METHODOLOGICAL FIX (Applied 2025-09-30):**

**The Problem:** Original implementation used `mean_vessel_capacity` from **trade patterns**, which varies by route and region. This created fundamental distortion in net balance because:
- **Demand side** used trade-pattern vessel capacity (could be smaller/larger than fleet average)
- **Supply side** counted actual fleet vessels uniformly
- Result: **Apples-to-oranges comparison** that artificially inflated/deflated net balance

**Example of Distortion:**
- Fleet average: 160,000 cbm per vessel (from 400 total vessels)
- Trade pattern average: 145,000 cbm (smaller vessels serving certain routes)
- Using 145,000 cbm → calculates need for 337.8 vessels
- Using 160,000 cbm → calculates need for 306.2 vessels
- **Difference: 31.6 vessels** (artificial tightness in balance)

**The Solution:** Use `average_size_cubic_meters` from **fleet data** instead of trade-based capacity. This ensures we compare:
- **Demand:** How many vessels from the actual fleet are needed
- **Supply:** How many vessels the actual fleet has
- **Result:** True net balance reflecting fleet availability vs requirements

**Component Breakdown:**

1. **Numerator Part 1:** `volume / (vessel_capacity × utilization_rate)`
   - Calculates number of **cargo deliveries** needed in the period
   - **vessel_capacity = fleet average capacity** (NOT trade pattern capacity)
   - Example: 1,000,000 MCM demand / (180,000 cbm × 0.85) = 6,536 deliveries

2. **Numerator Part 2:** `(laden_days + nonladen_days)`
   - Total **round-trip time** for a vessel
   - Example: 15 days laden + 10 days ballast = 25 days round-trip

3. **Denominator:** `days_in_period`
   - Converts round-trip time to **vessel-periods**
   - Example: 25 days / 30 days = 0.833 vessel-months per round-trip

**Complete Example:**
```python
volume = 1_000_000  # MCM
vessel_capacity = 180_000  # FLEET AVERAGE cubic meters (from ship_analysis_df)
utilization_rate = 0.85
laden_days = 15
nonladen_days = 10
days_in_period = 30  # monthly

ships_demand = (1_000_000 / (180_000 × 0.85)) × (15 + 10) / 30
             = 6,536 × 25 / 30
             = 5,447 vessel-months
             ≈ 181 vessels needed for the month
```

**Implementation (Regional Level):**

```python
# Step 1: Rename and prepare regional data (includes shipping_region column)
df_regional = df_shipping_demand.rename(columns={'Date': 'date'}).copy()

# Calculate days_in_period
df_regional['days_in_period'] = get_days_in_period(df_regional['date'], aggregation_level)

# CRITICAL FIX: Merge fleet average capacity data before calculating ships_demand
# This ensures we compare apples-to-apples: fleet capacity vs demand based on that same fleet capacity
df_regional = pd.merge(
    df_regional,
    ship_analysis_df[['date', 'average_size_cubic_meters']],
    on='date',
    how='left'
)

# Store trade-based capacity for reference/analysis, but use fleet capacity for demand calculation
df_regional['trade_vessel_capacity'] = df_regional['mean_vessel_capacity']
df_regional['mean_vessel_capacity'] = df_regional['average_size_cubic_meters']

# Step 2: Calculate ships_demand BY REGION using FLEET AVERAGE CAPACITY
historical_mask = df_regional['date'] <= hist_date_max

# Historical: use REALIZED utilization rates
hist_util = df_regional.loc[historical_mask, 'aggregate_utilization_rate'].fillna(0.75).clip(upper=0.98)

df_regional.loc[historical_mask, 'ships_demand'] = np.round(
    (df_regional.loc[historical_mask, 'value'] /
     (df_regional.loc[historical_mask, 'mean_vessel_capacity'] * hist_util)) *  # Uses fleet avg
    (df_regional.loc[historical_mask, 'mean_laden_days'] +
     df_regional.loc[historical_mask, 'mean_nonladen_days']) /
    df_regional.loc[historical_mask, 'days_in_period'], 1
)

# Future: use USER-SPECIFIED utilization rate
df_regional.loc[~historical_mask, 'ships_demand'] = np.round(
    (df_regional.loc[~historical_mask, 'value'] /
     (df_regional.loc[~historical_mask, 'mean_vessel_capacity'] * utilization_rate)) *  # Uses fleet avg
    (df_regional.loc[~historical_mask, 'mean_laden_days'] +
     df_regional.loc[~historical_mask, 'mean_nonladen_days']) /
    df_regional.loc[~historical_mask, 'days_in_period'], 1
)

# Step 3: Aggregate regional demands to GLOBAL
wm_avg = lambda x: np.average(x, weights=df_regional.loc[x.index, 'value'])
df_global = df_regional.groupby('date').agg({
    'value': 'sum',
    'ships_demand': 'sum',  # SUM of regional demands
    'mean_cargo_cubic_meters': wm_avg,
    'mean_laden_days': wm_avg,
    'mean_nonladen_days': wm_avg,
    'aggregate_utilization_rate': wm_avg,
    'mean_vessel_capacity': 'first',  # Fleet average (now same across all regions for a given date)
    'utilization_ratio': wm_avg,
    'days_in_period': 'first'
}).reset_index()

# Step 4: Merge global with fleet data
df_global = pd.merge(df_global, ship_analysis_df, on='date', how='left')
df_global['net'] = df_global['total_active_ships'] - df_global['ships_demand']
```

**Why This Fix Matters:**
- **Before:** Net balance reflected mismatch between trade patterns and fleet composition
- **After:** Net balance reflects true availability of the actual fleet to meet demand
- **Impact:** More accurate market tightness assessment, correct vessel count requirements

**Why Different Utilization Rates:**
- **Historical:** We know what actually happened → use realized rates
- **Future:** We're modeling scenarios → use analyst assumptions (default 85%)

#### 7.3 Net Balance Calculation

**Regional Balance (no fleet comparison):**
```python
# Regional balance shows demand by region but doesn't have net balance
# (fleet is global, not allocated by region)
df_regional: columns = [date, shipping_region, value, ships_demand, mean_cargo_cubic_meters, ...]
```

**Global Balance (with fleet comparison):**
```python
df_global['net'] = (
    df_global['total_active_ships'] -
    df_global['ships_demand']  # Sum of all regional demands
)
```

**Interpretation:**
- **Positive net:** Fleet surplus (more ships than needed globally)
- **Negative net:** Fleet deficit (shipping capacity shortage)
- **Near-zero:** Balanced market
- **Regional data:** Shows which regions drive demand (cannot compare to fleet at regional level)

#### 7.4 Utilization Ratio (Display Metric)

```python
df_global_shipping_balance['utilization_ratio'] = 0.0
df_global_shipping_balance.loc[historical_mask, 'utilization_ratio'] = hist_utilization * 100
df_global_shipping_balance.loc[~historical_mask, 'utilization_ratio'] = utilization_rate * 100
```
- Converted to percentage for user display
- Shows what utilization assumption was used for each period

**Final Output DataFrames:**

**Regional Balance:**
```
Columns: [
    'date', 'shipping_region', 'value', 'mean_cargo_cubic_meters', 'mean_laden_days',
    'mean_nonladen_days', 'aggregate_utilization_rate', 'mean_vessel_capacity',
    'days_in_period', 'ships_demand', 'utilization_ratio', 'year', 'quarter', 'month', 'season'
]
Note: No 'net' column (fleet is global, not regional)
```

**Global Balance:**
```
Columns: [
    'date', 'value', 'mean_cargo_cubic_meters', 'mean_laden_days', 'mean_nonladen_days',
    'aggregate_utilization_rate', 'mean_vessel_capacity', 'total_active_ships',
    'ships_added', 'ships_removed', 'average_size_cubic_meters', 'days_in_period',
    'ships_demand', 'net', 'utilization_ratio', 'year', 'quarter', 'month', 'season'
]
Note: 'ships_demand' = sum of regional demands; 'net' = total_active_ships - ships_demand
```

---

## 4. The Trade Analysis Engine: `kpler_analysis()`

**Location:** Lines 23-365

**Purpose:** Pre-aggregate Kpler trade data with volume-weighting for accuracy

### 4.1 Function Architecture

```python
def kpler_analysis(engine, start_date=None, end_date=None, aggregation_level='quarterly'):
    """
    Returns: (df_intracountry_trades, df_trades_shipping_region)
    """
```

### 4.2 Core SQL Query

**CTE Breakdown:**

```sql
WITH latest_upload AS (
    -- Identify most recent Kpler data snapshot
    SELECT MAX(upload_timestamp_utc) AS max_timestamp
    FROM at_lng.kpler_trades
),
supply_trades AS (
    -- Consolidate multi-destination cargoes to single supply record
    SELECT
        vessel_name,
        vessel_capacity_cubic_meters,
        start,
        origin_country_name,
        origin_location_name,
        MAX("end") AS "end",  -- Latest discharge date for multi-port cargoes
        SUM(cargo_origin_cubic_meters) AS cargo_origin_cubic_meters,
        SUM(cargo_destination_cubic_meters) AS cargo_destination_cubic_meters,
        SUM(mileage_nautical_miles) AS mileage_nautical_miles,
        SUM(ton_miles) AS ton_miles,
        COUNT(*) AS n_voyages  -- Number of discharge ports
    FROM at_lng.kpler_trades
    CROSS JOIN latest_upload
    WHERE upload_timestamp_utc = max_timestamp
        AND status = 'Delivered'
        [AND "end" >= %(start_date)s]  -- Optional date filter
        [AND "end" <= %(end_date)s]
    GROUP BY vessel_name, vessel_capacity_cubic_meters, start, origin_country_name, origin_location_name
)
SELECT DISTINCT
    a.vessel_name,
    c.vessel_type,  -- Mapped from capacity ranges
    a.start,
    a.origin_country_name,
    a.origin_location_name,
    a."end",
    b.destination_country_name,
    b.destination_location_name,
    a.vessel_capacity_cubic_meters,
    a.cargo_origin_cubic_meters,
    a.cargo_destination_cubic_meters,
    a.mileage_nautical_miles,
    a.ton_miles
FROM supply_trades a
INNER JOIN at_lng.kpler_trades b
    ON a.vessel_name = b.vessel_name
    AND a.start = b.start
    AND a."end" = b."end"
    AND b.status = 'Delivered'
LEFT JOIN at_lng.mapping_vessel_type_capacity c
    ON a.vessel_capacity_cubic_meters >= c.capacity_cubic_meters_min
    AND a.vessel_capacity_cubic_meters < c.capacity_cubic_meters_max
```

**Why This Complex Join:**
- Kpler data has one row per discharge port
- A vessel loading in Qatar may discharge in UK, France, and Italy → 3 rows with same origin
- Query consolidates to single "supply trade" with aggregated destination metrics

### 4.3 Ballast Leg Reconstruction

**Problem:** Kpler only records laden voyages. System needs non-laden (ballast) legs for round-trip calculation.

**Solution:** Synthetic ballast leg generation

```python
# Shift destination to next voyage's origin
df_trades['prev_end'] = df_trades.groupby(['vessel_name'])['end'].shift(1)
df_trades['prev_destination_country_name'] = df_trades.groupby(['vessel_name'])['destination_country_name'].shift(1)
df_trades['prev_destination_location_name'] = df_trades.groupby(['vessel_name'])['destination_location_name'].shift(1)

# Filter valid rows (where previous voyage exists)
mask = df_trades['prev_end'].notnull()

# Create synthetic ballast legs
new_rows = pd.DataFrame({
    'vessel_name': df_trades.loc[mask, 'vessel_name'],
    'vessel_type': df_trades.loc[mask, 'vessel_type'],
    'start': df_trades.loc[mask, 'prev_end'],  # Depart from previous destination
    'origin_country_name': df_trades.loc[mask, 'prev_destination_country_name'],
    'origin_location_name': df_trades.loc[mask, 'prev_destination_location_name'],
    'end': df_trades.loc[mask, 'start'],  # Arrive at next origin
    'destination_country_name': df_trades.loc[mask, 'origin_country_name'],
    'destination_location_name': df_trades.loc[mask, 'origin_location_name'],
    'status': 'non_laden',
    'vessel_capacity_cubic_meters': df_trades.loc[mask, 'prev_vessel_capacity_cubic_meters'],
    'cargo_origin_cubic_meters': 0,
    'cargo_destination_cubic_meters': 0
})

# Append ballast legs to trade data
df_trades = pd.concat([df_trades, new_rows])
```

**Example:**
```
Vessel: "LNG Queen"
Laden Voyage 1: Qatar → UK (Jan 1 - Jan 15)
Laden Voyage 2: Qatar → France (Jan 25 - Feb 10)

Reconstructed Ballast Leg:
Origin: UK
Destination: Qatar
Start: Jan 15 (end of Voyage 1)
End: Jan 25 (start of Voyage 2)
Status: non_laden
```

### 4.4 Data Cleaning & Enrichment

```python
# Calculate delivery duration
df_trades['delivery_days'] = (df_trades['end'] - df_trades['start']).dt.days
df_trades = df_trades[df_trades['delivery_days'] > 0]  # Remove invalid records

# Add temporal dimensions
df_trades['year'] = df_trades['end'].dt.year
df_trades['month'] = df_trades['end'].dt.month
df_trades['quarter'] = 'Q' + df_trades['end'].dt.quarter.astype(str)
df_trades['season'] = np.where(df_trades['month'].isin([10,11,12,1,2,3]), 'W', 'S')

# Cap cargo at vessel capacity (data quality check)
df_trades['cargo_origin_cubic_meters'] = df_trades[['cargo_origin_cubic_meters', 'vessel_capacity_cubic_meters']].min(axis=1)
df_trades['cargo_destination_cubic_meters'] = df_trades[['cargo_destination_cubic_meters', 'vessel_capacity_cubic_meters']].min(axis=1)

# Calculate utilization rates (ONLY for laden voyages)
laden_mask = df_trades['status'] == 'laden'
df_trades.loc[laden_mask, 'utilization_rate_destination'] = (
    df_trades.loc[laden_mask, 'cargo_destination_cubic_meters'] /
    df_trades.loc[laden_mask, 'vessel_capacity_cubic_meters']
)
df_trades.loc[laden_mask, 'utilization_rate_origin'] = (
    df_trades.loc[laden_mask, 'cargo_origin_cubic_meters'] /
    df_trades.loc[laden_mask, 'vessel_capacity_cubic_meters']
)

# Add shipping region classification
df_mapping_country = pd.read_sql(
    'SELECT DISTINCT country, shipping_region FROM at_lng.mappings_country',
    engine
)
df_trades = pd.merge(
    df_trades,
    df_mapping_country.rename(columns={'country': 'origin_country_name', 'shipping_region': 'origin_shipping_region'}),
    how='left', on='origin_country_name'
)
df_trades = pd.merge(
    df_trades,
    df_mapping_country.rename(columns={'country': 'destination_country_name', 'shipping_region': 'destination_shipping_region'}),
    how='left', on='destination_country_name'
)
```

### 4.5 Volume-Weighted Aggregation Logic

**Critical Technical Detail:** The system uses **cargo-weighted averages** for laden voyages and **capacity-weighted averages** for ballast legs.

#### 4.5.1 Intracountry Trade Aggregation

**Grouping Columns:**
```python
intracountry_group_cols = ['vessel_type', 'status'] + time_cols + [
    'origin_country_name', 'origin_shipping_region', 'destination_shipping_region'
]
```

**Laden Trade Aggregation:**
```python
df_intracountry_laden_agg = df_intracountry_laden.groupby(intracountry_group_cols).agg(
    # Volume-weighted delivery days
    weighted_delivery_days_sum=('delivery_days',
        lambda x: (x * df_intracountry_laden.loc[x.index, 'cargo_destination_cubic_meters']).sum()
    ),
    cargo_weight_sum=('cargo_destination_cubic_meters', 'sum'),

    # Raw sums for utilization calculation
    sum_cargo_destination=('cargo_destination_cubic_meters', 'sum'),
    sum_vessel_capacity=('vessel_capacity_cubic_meters', 'sum'),
    sum_cargo_origin=('cargo_origin_cubic_meters', 'sum'),

    # Additional metrics
    mean_mileage_nautical_miles=('mileage_nautical_miles', 'mean'),
    sum_ton_miles=('ton_miles', 'sum'),
    count_trades=('delivery_days', 'count')
).reset_index()

# Calculate volume-weighted mean
df_intracountry_laden_agg['mean_delivery_days'] = (
    df_intracountry_laden_agg['weighted_delivery_days_sum'] /
    df_intracountry_laden_agg['cargo_weight_sum']
).fillna(0)

# Calculate mean vessel capacity correctly (Version 2.1 fix)
df_intracountry_laden_agg['mean_vessel_capacity_cubic_meters'] = (
    df_intracountry_laden_agg['sum_vessel_capacity'] /
    df_intracountry_laden_agg['count_trades']
)
```

**Mathematical Formula:**

```
                    Σ(delivery_days_i × cargo_volume_i)
mean_delivery_days = ──────────────────────────────────
                              Σ(cargo_volume_i)
```

**Example:**
```python
Trade 1: 10 days, 150,000 cbm
Trade 2: 12 days, 180,000 cbm
Trade 3: 15 days, 160,000 cbm

weighted_sum = (10 × 150,000) + (12 × 180,000) + (15 × 160,000)
             = 1,500,000 + 2,160,000 + 2,400,000
             = 6,060,000

cargo_sum = 150,000 + 180,000 + 160,000 = 490,000

mean_delivery_days = 6,060,000 / 490,000 = 12.37 days

Simple average would be: (10 + 12 + 15) / 3 = 12.33 days
Volume-weighted gives more weight to larger cargo (12.37 > 12.33)
```

**Non-Laden Trade Aggregation:**
```python
df_intracountry_nonladen_agg = df_intracountry_nonladen.groupby(intracountry_group_cols).agg(
    # Capacity-weighted delivery days for ballast legs
    weighted_delivery_days_sum=('delivery_days',
        lambda x: (x * df_intracountry_nonladen.loc[x.index, 'vessel_capacity_cubic_meters']).sum()
    ),
    cargo_weight_sum=('vessel_capacity_cubic_meters', 'sum'),  # Use capacity as weight

    sum_cargo_destination=('cargo_destination_cubic_meters', 'sum'),  # Will be 0
    sum_vessel_capacity=('vessel_capacity_cubic_meters', 'sum'),
    sum_cargo_origin=('cargo_origin_cubic_meters', 'sum'),  # Will be 0

    mean_mileage_nautical_miles=('mileage_nautical_miles', 'mean'),
    sum_ton_miles=('ton_miles', 'sum'),
    count_trades=('delivery_days', 'count')
).reset_index()

# Calculate mean vessel capacity correctly (Version 2.1 fix)
df_intracountry_nonladen_agg['mean_vessel_capacity_cubic_meters'] = (
    df_intracountry_nonladen_agg['sum_vessel_capacity'] /
    df_intracountry_nonladen_agg['count_trades']
)
```

**Why Capacity-Weighting for Ballast:**
- Ballast legs carry no cargo (cargo = 0)
- Larger vessels take longer to reposition → should have more weight
- Using capacity as proxy for vessel "importance"

**Aggregate Utilization Rate Calculation:**

```python
# Portfolio-level utilization (NOT average of individual rates)
df_intracountry_trades['aggregate_utilization_rate_destination'] = (
    df_intracountry_trades['sum_cargo_destination'] /
    df_intracountry_trades['sum_vessel_capacity']
).clip(0, 1)

df_intracountry_trades['aggregate_utilization_rate_origin'] = (
    df_intracountry_trades['sum_cargo_origin'] /
    df_intracountry_trades['sum_vessel_capacity']
).clip(0, 1)
```

**Why Aggregate (Not Average):**

**Wrong Approach:**
```python
# This would be WRONG
mean_utilization = df['utilization_rate'].mean()
```

**Problem:** Treats all voyages equally regardless of vessel size

**Correct Approach:**
```python
aggregate_utilization = total_cargo / total_capacity
```

**Example:**
```
Trade 1: 150,000 cbm cargo in 180,000 cbm vessel = 83% utilization
Trade 2: 140,000 cbm cargo in 160,000 cbm vessel = 87.5% utilization
Trade 3: 100,000 cbm cargo in 200,000 cbm vessel = 50% utilization

Wrong (simple average): (83% + 87.5% + 50%) / 3 = 73.5%

Correct (aggregate):
    (150,000 + 140,000 + 100,000) / (180,000 + 160,000 + 200,000)
    = 390,000 / 540,000
    = 72.2%

The low-utilization large vessel (Trade 3) correctly pulls down the aggregate.
```

#### 4.5.2 Shipping Region Trade Aggregation

**Same logic as intracountry, but grouped by:**
```python
shipping_group_cols = ['vessel_type', 'status'] + time_cols + [
    'origin_shipping_region', 'destination_shipping_region'
]
```

**Output DataFrames:**

**1. Intracountry Trades:**
```
Columns: [
    'vessel_type', 'status', ...time_cols..., 'origin_country_name',
    'origin_shipping_region', 'destination_shipping_region',
    'sum_cargo_destination', 'sum_vessel_capacity', 'sum_cargo_origin',
    'mean_delivery_days', 'mean_vessel_capacity_cubic_meters',
    'aggregate_utilization_rate_destination', 'aggregate_utilization_rate_origin',
    'mean_cargo_destination_cubic_meters', 'mean_cargo_origin_cubic_meters',
    'mean_mileage_nautical_miles', 'sum_ton_miles', 'count_trades'
]
```

**2. Shipping Region Trades:**
```
Columns: [
    'vessel_type', 'status', ...time_cols...,
    'origin_shipping_region', 'destination_shipping_region',
    'sum_cargo_destination', 'sum_vessel_capacity', 'sum_cargo_origin',
    'mean_delivery_days', 'mean_vessel_capacity_cubic_meters',
    'aggregate_utilization_rate_destination', 'aggregate_utilization_rate_origin',
    'weighted_utilization_rate_destination', 'weighted_utilization_rate_origin',
    'weighted_vessel_capacity_destination', 'weighted_vessel_capacity_origin',
    'mean_cargo_destination_cubic_meters', 'mean_cargo_origin_cubic_meters',
    'mean_mileage_nautical_miles', 'sum_ton_miles', 'count_trades'
]
```

**Backward Compatibility Columns:**
```python
df_trades_shipping_region['weighted_utilization_rate_destination'] = df_trades_shipping_region['aggregate_utilization_rate_destination']
df_trades_shipping_region['weighted_utilization_rate_origin'] = df_trades_shipping_region['aggregate_utilization_rate_origin']
df_trades_shipping_region['weighted_vessel_capacity_destination'] = df_trades_shipping_region['mean_vessel_capacity_cubic_meters']
df_trades_shipping_region['weighted_vessel_capacity_origin'] = df_trades_shipping_region['mean_vessel_capacity_cubic_meters']
```
- Preserves old naming convention for downstream code
- "weighted" and "aggregate" mean the same thing in this context

---

## 5. Demand View vs. Supply View: The Critical Distinction

### 5.1 Conceptual Framework

| Aspect | Demand View | Supply View |
|--------|------------|-------------|
| **Perspective** | Import terminal operator | Export terminal operator |
| **Primary Question** | "How many ships do I need to receive LNG?" | "How many ships do I need to export LNG?" |
| **Laden Leg Focus** | **Destination** (where cargo unloads) | **Origin** (where cargo loads) |
| **Non-Laden Leg Focus** | **Origin** (where ballast leg departs) | **Destination** (where ballast leg arrives) |
| **Cargo Metric** | `cargo_destination_cubic_meters` | `cargo_origin_cubic_meters` |
| **Utilization Metric** | `aggregate_utilization_rate_destination` | `aggregate_utilization_rate_origin` |

### 5.2 Why This Matters: A Worked Example

**Scenario:**
- Qatar exports 10 BCM/year to UK
- UK also receives 5 BCM/year from US
- Average vessel capacity: 180,000 cbm
- Utilization: 85%
- Laden leg: 15 days
- Ballast leg: 10 days

#### Demand View (UK Import Terminal Perspective):

**Question:** How many vessels does UK need to receive total imports?

**Calculation:**
```python
# Total UK imports
total_imports = 10 BCM (Qatar) + 5 BCM (US) = 15 BCM

# Laden legs measured at UK (destination)
# These are the cargo arrivals
laden_deliveries = 15 BCM / (180,000 cbm × 0.85) = 98,039 deliveries

# Non-laden legs measured at Qatar/US (origins)
# These are the vessels departing EMPTY from UK back to Qatar/US
# The system tracks where the ballast leg STARTS (UK) using origin_shipping_region

# Ships demand
ships_demand = 98,039 × (15 + 10) / 365 = 6,714 vessel-years ≈ 18.4 vessels
```

**Result:** UK needs ~18 vessels continuously arriving with LNG

#### Supply View (Qatar Export Terminal Perspective):

**Question:** How many vessels does Qatar need to export LNG?

**Calculation:**
```python
# Qatar exports
total_exports = 10 BCM (to UK)

# Laden legs measured at Qatar (origin)
# These are the cargo loadings
laden_loadings = 10 BCM / (180,000 cbm × 0.85) = 65,359 loadings

# Non-laden legs measured at UK (destination)
# These are the vessels arriving EMPTY at Qatar from UK
# The system tracks where the ballast leg ENDS (Qatar) using destination_shipping_region

# Ships demand
ships_demand = 65,359 × (15 + 10) / 365 = 4,476 vessel-years ≈ 12.3 vessels
```

**Result:** Qatar needs ~12 vessels continuously loading LNG

### 5.3 Implementation in Code

The implementation directly uses the appropriate columns based on the view type:

**Supply View:**
- Laden region: `origin_shipping_region`
- Non-laden region: `destination_shipping_region`
- Cargo column: `median_cargo_origin_cubic_meters`
- Utilization column: `mean_utilization_rate_origin`

**Demand View:**
- Laden region: `destination_shipping_region`
- Non-laden region: `origin_shipping_region`
- Cargo column: `median_cargo_destination_cubic_meters`
- Utilization column: `mean_utilization_rate_destination`

---

## 6. Aggregation Levels: Technical Implementation

### 6.1 Monthly Aggregation

**Time Columns:** `['year', 'month']`

**Period Length:** 28-31 days (varies by month and leap year)

**Date Formatting:** `'YYYY-MM'`

**Example:**
```
2024-01-01 → January 2024 (31 days)
2024-02-01 → February 2024 (29 days, leap year)
2024-03-01 → March 2024 (31 days)
```

### 6.2 Quarterly Aggregation

**Time Columns:** `['year', 'quarter']`

**Period Length:** 90-92 days

**Date Formatting:** `'YYYY Q1'`, `'YYYY Q2'`, etc.

**Quarter Definitions:**
```
Q1: January 1 - March 31 (90 days, 91 in leap years)
Q2: April 1 - June 30 (91 days)
Q3: July 1 - September 30 (92 days)
Q4: October 1 - December 31 (92 days)
```

**Example:**
```python
date = pd.Timestamp('2024-02-15')
quarter = date.dt.quarter  # 1
quarter_label = f'Q{quarter}'  # 'Q1'
```

### 6.3 Seasonal Aggregation

**Time Columns:** `['year', 'season']`

**Period Length:**
- Winter (W): 182 days (Oct 1 - Mar 31)
- Summer (S): 183 days (Apr 1 - Sep 30)

**Season Definition Logic:**
```python
df['season'] = np.where(
    df['month'].isin([10, 11, 12, 1, 2, 3]),
    'W',  # Winter
    'S'   # Summer
)
```

**Date Formatting:**
```
2024-W → Winter 2024 (Oct 2023 - Mar 2024)
2024-S → Summer 2024 (Apr 2024 - Sep 2024)
```

**SQL Date Calculation:**
```sql
CASE
    WHEN EXTRACT(MONTH FROM start_date::DATE) IN (10, 11, 12, 1, 2, 3) THEN
        DATE_TRUNC('year', start_date::DATE)  -- Winter starts at year boundary
    ELSE
        DATE_TRUNC('year', start_date::DATE) + INTERVAL '6 months'  -- Summer is mid-year
END
```

### 6.4 Yearly Aggregation

**Time Columns:** `['year']`

**Period Length:** 365 days (366 in leap years)

**Date Formatting:** `'YYYY'`

**Leap Year Detection:**
```python
year = df['date'].dt.year
days_in_year = year.map(
    lambda y: 366 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 365
)
```

---

## 7. Visualization Layer: Dashboard Implementation

**Location:** `dash_shipping_lng_snd/pages/shipping_balance.py`

### 7.1 Page Layout Structure

```
┌─────────────────────────────────────────────────────────────────┐
│ SECTION 1: Global Shipping Balance Overview                     │
├─────────────────────────────────────────────────────────────────┤
│ Controls: [Aggregation] [Vessel Age] [Utilization %] [Window]  │
│                                                                  │
│ ┌───────────────────────┬───────────────────────────┐           │
│ │  Demand View Chart    │  Supply View Chart        │           │
│ │  (Import perspective) │  (Export perspective)     │           │
│ │                       │                           │           │
│ │  - Total Active Ships │  - Total Active Ships     │           │
│ │  - Ships Demand Total │  - Ships Supply Total     │           │
│ │  - Net Balance (bars) │  - Net Balance (bars)     │           │
│ └───────────────────────┴───────────────────────────┘           │
│                                                                  │
│ ┌───────────────────────┬───────────────────────────┐           │
│ │  Demand Metrics Table │  Supply Metrics Table     │           │
│ │  (Date, Ships, Net,   │  (Date, Ships, Net,       │           │
│ │   Utilization, Volume)│   Utilization, Volume)    │           │
│ └───────────────────────┴───────────────────────────┘           │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ SECTION 2: Trade Analysis by Region Pairs                       │
├─────────────────────────────────────────────────────────────────┤
│ Controls: [Metric] [Origin Region] [Year] [Status]             │
│                                                                  │
│ ┌───────────────────────────────────────────────────┐           │
│ │  Interactive DataTable                             │           │
│ │  Rows: Origin→Destination Region Pairs            │           │
│ │  Columns: Vessel Types (Standard, DFDE, MEGI...)  │           │
│ │  Values: Count/TonMiles/Delivery Days/etc.        │           │
│ └───────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ SECTION 3: Intracountry Trade Analysis                          │
├─────────────────────────────────────────────────────────────────┤
│ Controls: [Year] [Status]                                       │
│                                                                  │
│ ┌───────────────────────┬───────────────────────────┐           │
│ │  Trade Count Chart    │  Ton Miles Chart          │           │
│ │  (Stacked by country) │  (Stacked by country)     │           │
│ └───────────────────────┴───────────────────────────┘           │
│                                                                  │
│ ┌───────────────────────┬───────────────────────────┐           │
│ │  Count Data Table     │  Ton Miles Data Table     │           │
│ └───────────────────────┴───────────────────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 Data Flow in Dashboard

**User Action → Callback Chain:**

```
User clicks "Refresh Data" button
            ↓
refresh_data() callback triggers
            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 1. Get date range for window picker                             │
│ 2. Call global_shipping_balance(lng_view='demand')              │
│ 3. Call global_shipping_balance(lng_view='supply')              │
│ 4. Call kpler_analysis() for trade data                         │
│ 5. Build dropdown options                                       │
│ 6. Store all data in dcc.Store components                       │
└─────────────────────────────────────────────────────────────────┘
            ↓
update_visualizations() callback triggers
            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 1. Load data from stores                                        │
│ 2. Create Demand View chart (Plotly)                            │
│ 3. Create Supply View chart (Plotly)                            │
│ 4. Format tables for demand/supply metrics                      │
│ 5. Populate dropdown options                                    │
└─────────────────────────────────────────────────────────────────┘
```

### 7.3 Key Callbacks

#### 7.3.1 `refresh_data()` - Master Data Fetch

**Location:** Lines 734-881

**Triggers:**
- User clicks "Refresh Data" button
- User changes aggregation level
- User changes vessel age
- User changes utilization rate
- User changes window end date

**Process:**
```python
def refresh_data(n_clicks, aggregation_level='monthly', vessel_age=20,
                 utilization_rate=85, window_end_date=None):

    # 1. Parse and validate window_end_date
    if window_end_date and isinstance(window_end_date, str) and window_end_date.strip():
        window_end_date = pd.to_datetime(window_end_date)
    else:
        # Default to yesterday
        window_end_date = pd.Timestamp.now() - pd.Timedelta(days=1)

    # 2. Get date range from database for picker limits
    query_date_range = '''
        SELECT MIN("end") as min_date, MAX("end") as max_date
        FROM at_lng.kpler_trades
        WHERE status = 'Delivered' AND "end" IS NOT NULL
    '''
    date_range = pd.read_sql(query_date_range, engine)

    # 3. Convert utilization % to decimal
    utilization_rate_decimal = utilization_rate / 100.0

    # 4. Fetch both demand and supply views
    df_global_shipping_balance = global_shipping_balance(
        aggregation_level, vessel_age,
        lng_view='demand',
        utilization_rate=utilization_rate_decimal,
        window_end_date=window_end_date
    )

    df_global_shipping_balance_supply = global_shipping_balance(
        aggregation_level, vessel_age,
        lng_view='supply',
        utilization_rate=utilization_rate_decimal,
        window_end_date=window_end_date
    )

    # 5. Get historical date max
    query_max_hist_date = '''
        SELECT MAX("end") as hist_date_max
        FROM at_lng.kpler_trades
        WHERE status='Delivered'
        AND upload_timestamp_utc = (SELECT MAX(upload_timestamp_utc) FROM at_lng.kpler_trades)
    '''
    hist_date_max_df = pd.read_sql(query_max_hist_date, engine)
    max_date = pd.to_datetime(hist_date_max_df.hist_date_max.dt.date[0])
    hist_date_max = max_date + pd.offsets.MonthEnd(0)

    # 6. Fetch trade data (limited to historical only)
    df_intracountry_trades, df_trades_shipping_region = kpler_analysis(
        engine, end_date=hist_date_max, aggregation_level=aggregation_level
    )

    # 7. Build dropdown options
    origin_regions = sorted(df_trades_shipping_region['origin_shipping_region'].unique())
    years = sorted(df_trades_shipping_region['year'].unique())
    statuses = sorted(df_trades_shipping_region['status'].unique())

    # 8. Convert to JSON and store
    shipping_data = df_trades_shipping_region.to_json(date_format='iso', orient='split')
    shipping_balance = df_global_shipping_balance.to_json(date_format='iso', orient='split')
    shipping_balance_supply = df_global_shipping_balance_supply.to_json(date_format='iso', orient='split')
    intracountry_data = df_intracountry_trades.to_json(date_format='iso', orient='split')

    # 9. Create window info text
    window_start_dt = window_end_date - pd.Timedelta(days=365)
    window_info = f"📊 Using patterns from {window_start_dt.strftime('%Y-%m-%d')} to {window_end_date.strftime('%Y-%m-%d')} for future projections"

    return (shipping_data, shipping_balance, shipping_balance_supply,
            options_data, refresh_timestamp, intracountry_data,
            window_info, placeholder_text, min_date_str, window_end_date_str)
```

#### 7.3.2 `update_visualizations()` - Chart Generation

**Location:** Lines 897-1356

**Process:**
```python
def update_visualizations(shipping_data, shipping_balance, shipping_balance_supply, ...):

    # 1. Load data from JSON stores
    df_trades_shipping_region = pd.read_json(StringIO(shipping_data), orient='split')
    df_global_shipping_balance = pd.read_json(StringIO(shipping_balance), orient='split')
    df_global_shipping_balance_supply = pd.read_json(StringIO(shipping_balance_supply), orient='split')

    # 2. Create Demand View chart
    fig_global_shipping = make_subplots(specs=[[{"secondary_y": True}]])

    # Primary Y-axis: Number of ships
    fig_global_shipping.add_trace(
        go.Scatter(
            x=df_global_shipping_balance['date'],
            y=df_global_shipping_balance['total_active_ships'],
            name='Total Active Ships',
            mode='lines+markers',
            line=dict(color='#2E86C1', width=2)
        ),
        secondary_y=False
    )

    fig_global_shipping.add_trace(
        go.Scatter(
            x=df_global_shipping_balance['date'],
            y=df_global_shipping_balance['ships_demand'],
            name='Ships Demand Total',
            mode='lines+markers',
            line=dict(color='#22c55e', width=2)
        ),
        secondary_y=False
    )

    # Secondary Y-axis: Net balance
    fig_global_shipping.add_trace(
        go.Bar(
            x=df_global_shipping_balance['date'],
            y=df_global_shipping_balance['net'],
            name='Net New',
            marker_color='rgba(239, 68, 68, 0.5)'
        ),
        secondary_y=True
    )

    # 3. Format X-axis based on aggregation level
    if aggregation_level == 'monthly':
        tick_format = '%b %Y'  # Jan 2024
        dtick = "M3"  # Every 3 months
    elif aggregation_level == 'quarterly':
        tick_format = '%Y Q%q'  # 2024 Q1
        dtick = "M3"
    elif aggregation_level == 'seasonal':
        tick_format = '%Y %b'  # 2024 Jan (for winter), 2024 Jul (for summer)
        dtick = "M6"
    else:  # yearly
        tick_format = '%Y'  # 2024
        dtick = "M12"

    fig_global_shipping.update_xaxes(
        tickformat=tick_format,
        dtick=dtick,
        tickmode='auto',
        nticks=15
    )

    # 4. Create Supply View chart (same structure, different data)
    # ... (similar code for fig_global_shipping_supply)

    # 5. Create data tables
    demand_table_data = df_global_shipping_balance[[
        'date', 'total_active_ships', 'ships_demand', 'net',
        'utilization_ratio', 'value'
    ]].copy()

    # Format date column based on aggregation level
    if aggregation_level == 'quarterly':
        demand_table_data['date'] = demand_table_data['date'].dt.to_period('Q').astype(str)
    elif aggregation_level == 'seasonal':
        demand_table_data['date'] = demand_table_data['date'].apply(
            lambda x: f"{x.year}-{'W' if x.month == 1 else 'S'}"
        )
    elif aggregation_level == 'yearly':
        demand_table_data['date'] = demand_table_data['date'].dt.year
    else:  # monthly
        demand_table_data['date'] = demand_table_data['date'].dt.strftime('%Y-%m')

    demand_table = dash_table.DataTable(
        columns=[...],
        data=demand_table_data.to_dict('records'),
        style_table={'overflowX': 'auto'},
        export_format='xlsx'
    )

    return (fig_global_shipping, fig_global_shipping_supply,
            region_options, year_options, selected_year,
            status_options, selected_statuses,
            demand_table, supply_table)
```

### 7.4 Chart Styling Standards

**Professional Color Palette:**
```python
# Primary colors
McKinsey_Blue = '#2E86C1'
Professional_Green = '#22c55e'
Professional_Red = '#ef4444'
Professional_Yellow = '#F7DC6F'

# Background colors
Plot_Background = 'rgba(248, 249, 250, 0.5)'
Paper_Background = 'white'
Grid_Color = 'rgba(200, 200, 200, 0.3)'
```

**Layout Standards:**
```python
fig.update_layout(
    plot_bgcolor='rgba(248, 249, 250, 0.5)',
    paper_bgcolor='white',
    margin=dict(l=70, r=70, t=80, b=180),
    height=600,
    hovermode='x unified',
    hoverlabel=dict(
        bgcolor='rgba(255, 255, 255, 0.95)',
        bordercolor='rgba(200, 200, 200, 0.8)',
        font=dict(size=11, color='#2C3E50')
    )
)
```

---

## 8. Data Quality & Edge Cases

### 8.1 Data Quality Checks

**1. Cargo Volume Capping**
```python
# Physical impossibility: cargo cannot exceed vessel capacity
df_trades['cargo_origin_cubic_meters'] = df_trades[[
    'cargo_origin_cubic_meters',
    'vessel_capacity_cubic_meters'
]].min(axis=1)
```

**2. Invalid Delivery Days**
```python
# Transit time must be positive
df_trades['delivery_days'] = (df_trades['end'] - df_trades['start']).dt.days
df_trades = df_trades[df_trades['delivery_days'] > 0]
```

**3. Utilization Rate Clipping**
```python
# Utilization cannot exceed 98% (leave margin for measurement error)
hist_utilization = hist_utilization.fillna(0.75).clip(upper=0.98)
```

**4. Missing Data Handling**
```python
# Drop rows missing essential metrics
essential_cols = ['mean_laden_days', 'mean_nonladen_days', 'mean_cargo_cubic_meters',
                  'aggregate_utilization_rate', 'mean_vessel_capacity']
df = df.dropna(subset=essential_cols, how='any')
```

### 8.2 Edge Cases Handled

**1. Partial Month Data**
- Historical cutoff uses `pd.offsets.MonthEnd(0)` to ensure complete months
- Prevents partial-month artifacts in transition from historical to forecast

**2. Leap Years**
- Days in period calculation accounts for leap years
- 2024 Feb has 29 days, 2025 Feb has 28 days

**3. Vessel Retirement Clustering**
- Retirement distribution algorithm spreads dates across 12 months
- Prevents artificial spikes from mass retirements in single month

**4. Multi-Port Discharge Cargoes**
- `supply_trades` CTE consolidates multi-destination voyages
- Sums cargo across all discharge ports for single loading event

**5. Missing Shipping Region Mappings**
- Left join with mapping table preserves unmapped countries
- Downstream filtering removes rows without region classification

**6. Zero-Volume Periods**
- Aggregation functions use `.fillna(0)` for continuity
- Prevents NaN propagation in calculations

---

## 9. Performance Optimizations

### 9.1 Database Query Optimization

**1. Latest Upload Timestamp CTE**
```sql
WITH latest_upload AS (
    SELECT MAX(upload_timestamp_utc) AS max_timestamp
    FROM at_lng.kpler_trades
)
```
- Calculates once, reused in all subsequent CTEs
- Prevents repeated `MAX()` subquery execution

**2. Aggregation at Source**
- Volume-weighting performed in `kpler_analysis()` once
- Downstream code uses pre-aggregated metrics
- Eliminates need for repeated weighted calculations

**3. Date Filtering in SQL**
```sql
WHERE "end" >= %(start_date)s AND "end" <= %(end_date)s
```
- Reduces rows returned from database
- PostgreSQL can use indexes on date columns

### 9.2 Memory Management

**1. DataFrame Chunking in Callbacks**
```python
# Convert to JSON for Dash storage (reduces memory footprint)
shipping_data = df_trades_shipping_region.to_json(date_format='iso', orient='split')
```

**2. Session Storage**
```python
dcc.Store(id='trades-shipping-data-store', storage_type='session')
```
- Stores data in browser session storage
- Cleared when user closes browser
- Prevents memory leaks from stale data

**3. Selective Column Retention**
```python
# Only keep columns needed downstream
laden_cols = ['shipping_region'] + time_cols + ['mean_laden_days']
laden_metrics = laden_metrics[[c for c in laden_cols if c in laden_metrics.columns]]
```

### 9.3 Computational Efficiency

**1. Vectorized Operations**
```python
# Use pandas/numpy vectorization instead of loops
df['season'] = np.where(df['month'].isin([10,11,12,1,2,3]), 'W', 'S')
```

**2. Group-Apply Pattern**
```python
# Single groupby with multiple aggregations
df.groupby(group_cols).agg({
    'mean_delivery_days': 'first',
    'sum_cargo_origin': 'sum',
    'count_trades': 'sum'
})
```

**3. Merge Strategy**
```python
# Use inner joins where possible (faster than left joins)
# Left joins only when data may be missing
```

---

## 10. Critical Formulas Summary

### 10.1 Ships Demand Calculation

**Formula:**
```
ships_demand = (volume / (vessel_capacity × utilization_rate)) ×
               (laden_days + nonladen_days) / days_in_period
```

**Units:**
- `volume`: cubic meters (m³) for the period
- `vessel_capacity`: cubic meters (cbm)
- `utilization_rate`: dimensionless (0.0-1.0)
- `laden_days`: days
- `nonladen_days`: days
- `days_in_period`: days (period length)
- **Result:** number of vessels needed for the period

### 10.2 Volume-Weighted Average

**Formula:**
```
weighted_metric = Σ(metric_i × cargo_i) / Σ(cargo_i)
```

**Example (Delivery Days):**
```
mean_delivery_days = Σ(delivery_days_i × cargo_volume_i) / Σ(cargo_volume_i)
```

### 10.3 Aggregate Utilization Rate

**Formula:**
```
aggregate_utilization = Σ(cargo_i) / Σ(capacity_i)
```

**NOT:**
```
mean(cargo_i / capacity_i)  # This would be wrong
```

### 10.4 Net Balance

**Formula:**
```
net = total_active_ships - ships_demand
```

**Interpretation:**
- `net > 0`: Fleet surplus
- `net < 0`: Fleet deficit
- `net ≈ 0`: Balanced market

---

## 11. Common Analyst Questions Answered

### Q1: Why do Demand and Supply views show different ship counts?

**Answer:** They measure vessels from different perspectives:
- **Demand View:** Counts vessels arriving at import terminals (cargo delivery endpoint)
- **Supply View:** Counts vessels departing from export terminals (cargo loading endpoint)

A vessel delivering to Europe from Qatar appears in:
- Europe's demand view (destination)
- Qatar's supply view (origin)

Global totals may differ because:
- Regional focus varies (not all regions import/export equally)
- Ballast leg directionality affects regional assignment

### Q2: Why does changing the window end date affect future forecasts but not history?

**Answer:**
- **Historical periods:** Use actual realized trade patterns from all available history
- **Future periods:** Use patterns from the 365-day window to project forward

Window purpose: Allows choosing which recent patterns inform forecasts (e.g., use 2024 patterns vs. 2023 patterns)

### Q3: Why are utilization rates different in historical vs. forecast periods?

**Answer:**
- **Historical:** System uses **realized utilization rates** from actual trade data
  - These vary by period, region, and vessel type
  - Reflects actual market efficiency
- **Future:** System uses **user-specified utilization rate** (default 85%)
  - Allows scenario modeling: "What if we achieve 90% utilization?"
  - Industry standard assumption: 80-85%

### Q4: How does the system handle seasonal patterns?

**Answer:**
- Seasonal aggregation: Winter (Oct-Mar) vs. Summer (Apr-Sep)
- Volume-weighted metrics capture seasonal variations:
  - More laden voyages in winter → higher winter delivery day weights
  - Different utilization rates by season
- User can select seasonal aggregation to see explicit seasonal balance

### Q5: Why might ships_demand be non-integer?

**Answer:** Ships demand represents **vessel-periods**, not discrete vessels.

Example (Monthly):
- 12.3 ships needed per month = One vessel making 12.3 round-trips in a month
- Or equivalently: 12 vessels fully utilized + 1 vessel at 30% utilization

Fractional values are mathematically correct and reflect partial vessel utilization.

### Q6: What happens if a region has no historical trade data?

**Answer:**
- For **historical periods:** Row is dropped (no data = no calculation possible)
- For **future periods (Supply View only):** Fallback to global averages
  - Uses mean of window metrics
  - Default utilization: 75%
- **Demand View:** Less prone to missing data (more import terminals than export)

### Q7: How accurate are the vessel retirement projections?

**Answer:**
Retirement dates use:
1. **Build year + life expectancy (default 20 years)**
2. **Distribution across 12 months** to prevent clustering
3. **Max(today, theoretical_retirement)** to prevent backdating

**Limitations:**
- Assumes uniform retirement timing (reality: early scrapping, life extensions)
- Does not account for conversions (LNG→FSRU)
- Distribution algorithm is simplistic (linear spread)

**Best practice:** Sensitivity analysis with life expectancy 15-25 years

### Q8: Why do aggregate utilization rates differ from simple averages?

**Answer:**
**Simple average:** Treats all voyages equally
```
mean_utilization = mean(cargo_i / capacity_i)
```

**Aggregate rate:** Weights by vessel capacity
```
aggregate_utilization = sum(cargo_i) / sum(capacity_i)
```

**Why aggregate is correct:**
- Larger vessels should have more influence on fleet-level utilization
- A 200,000 cbm vessel at 50% has more impact than a 150,000 cbm vessel at 90%
- Aggregate rate reflects portfolio-level efficiency

**Example:**
```
Vessel 1: 150,000 cbm capacity, 120,000 cbm cargo → 80% utilization
Vessel 2: 180,000 cbm capacity, 150,000 cbm cargo → 83% utilization
Vessel 3: 200,000 cbm capacity,  80,000 cbm cargo → 40% utilization

Simple average: (80% + 83% + 40%) / 3 = 67.7%
Aggregate: (120k + 150k + 80k) / (150k + 180k + 200k) = 350k / 530k = 66.0%

Aggregate correctly gives more weight to the large underutilized vessel.
```

---

## 12. Troubleshooting Guide

### Issue 1: Charts show no data

**Possible Causes:**
1. Database connection failure
2. Empty Kpler trades table
3. Date filter excluding all data

**Diagnostic Steps:**
```python
# Check database connection
try:
    pd.read_sql("SELECT 1", engine)
    print("✓ Database connected")
except Exception as e:
    print(f"✗ Database error: {e}")

# Check Kpler data availability
row_count = pd.read_sql(
    "SELECT COUNT(*) FROM at_lng.kpler_trades WHERE status='Delivered'",
    engine
).iloc[0, 0]
print(f"Kpler trades: {row_count} rows")

# Check date range
date_range = pd.read_sql(
    "SELECT MIN(end), MAX(end) FROM at_lng.kpler_trades WHERE status='Delivered'",
    engine
)
print(f"Date range: {date_range.iloc[0, 0]} to {date_range.iloc[0, 1]}")
```

### Issue 2: Ships demand shows sudden spikes

**Possible Causes:**
1. Incorrect days_in_period calculation
2. Missing vessel capacity data
3. Extreme utilization rates

**Diagnostic Steps:**
```python
# Check days_in_period
df_check = df_global_shipping_balance[['date', 'days_in_period', 'ships_demand']]
print(df_check[df_check['ships_demand'] > df_check['ships_demand'].quantile(0.95)])

# Check for missing capacity
print(df_shipping_demand['mean_vessel_capacity'].describe())
print(df_shipping_demand['mean_vessel_capacity'].isna().sum())

# Check utilization rates
print(df_shipping_demand['aggregate_utilization_rate'].describe())
```

### Issue 3: Demand and Supply views identical

**Possible Causes:**
1. `lng_view` parameter not being passed correctly
2. Data aggregation issue in `kpler_analysis()`

**Diagnostic Steps:**
```python
# Verify view parameter is being passed correctly
print(f"lng_view parameter: {lng_view}")

# Check column differences
demand_cols = set(df_demand.columns)
supply_cols = set(df_supply.columns)
print(f"Common columns: {demand_cols & supply_cols}")
print(f"Demand-only columns: {demand_cols - supply_cols}")
print(f"Supply-only columns: {supply_cols - demand_cols}")
```

### Issue 4: Historical/forecast transition shows discontinuity

**Possible Causes:**
1. Different utilization rates at boundary
2. Missing window data for forecast period
3. Aggregation level mismatch

**Diagnostic Steps:**
```python
# Check transition point
hist_max = df_global_shipping_balance[
    df_global_shipping_balance['date'] <= hist_date_max
]['date'].max()
forecast_min = df_global_shipping_balance[
    df_global_shipping_balance['date'] > hist_date_max
]['date'].min()

transition_data = df_global_shipping_balance[
    (df_global_shipping_balance['date'] >= hist_max - pd.Timedelta(days=90)) &
    (df_global_shipping_balance['date'] <= forecast_min + pd.Timedelta(days=90))
][['date', 'ships_demand', 'utilization_ratio', 'mean_laden_days']]

print(transition_data)
```

---

## 13. Future Enhancement Opportunities

### 13.1 Technical Enhancements

1. **Dynamic Vessel Type Segmentation**
   - Currently aggregates across all vessel types
   - Could segment by vessel type (DFDE, MEGI, Standard) for type-specific balance

2. **Canal Capacity Constraints**
   - Incorporate Panama Canal, Suez Canal capacity limits
   - Model impact of canal closures on routing and delivery times

3. **Port Congestion Modeling**
   - Add waiting time at terminals to delivery days
   - Dynamic adjustment based on port utilization

4. **Spot vs. Contract Differentiation**
   - Separate balance for spot market vs. contracted fleet
   - Model impact of contract expirations on availability

### 13.2 Analytical Enhancements

1. **Probabilistic Forecasting**
   - Monte Carlo simulation for utilization rates
   - Confidence intervals for ships_demand

2. **Scenario Management**
   - Save/load user-defined scenarios
   - Compare multiple scenarios side-by-side

3. **Sensitivity Analysis Dashboard**
   - Tornado diagrams for parameter sensitivity
   - Automated identification of high-impact variables

4. **Regional Drill-Down**
   - Click region in chart → see regional details
   - Regional supply/demand balance sub-views

### 13.3 Data Integration Enhancements

1. **Real-Time AIS Integration**
   - Live vessel positioning for current on-water calculations
   - Actual vs. projected delivery time comparison

2. **Forward Curve Integration**
   - LNG price curves from market data
   - Correlation analysis: shipping availability vs. price

3. **Weather Impact Modeling**
   - Historical weather-related delays
   - Seasonal adjustment for storm patterns

4. **Fleet Orderbook Granularity**
   - Shipyard-specific delivery schedules
   - Cancellation risk modeling

---

## 14. Glossary of Technical Terms

| Term | Definition |
|------|------------|
| **Aggregate Utilization** | Portfolio-level cargo/capacity ratio: Σ(cargo) / Σ(capacity) |
| **Ballast Leg** | Non-laden voyage segment with no cargo (vessel repositioning) |
| **CBM** | Cubic meters (vessel capacity unit) |
| **Days in Period** | Number of days in the aggregation period (28-366 depending on level) |
| **Demand View** | Import-centric perspective measuring vessels at discharge ports |
| **DFDE** | Dual Fuel Diesel Electric (vessel propulsion type) |
| **FSU** | Floating Storage Unit (excluded from active fleet) |
| **Hist Date Max** | Cutoff date between realized history and forecast periods |
| **Laden Leg** | Cargo-carrying voyage segment |
| **Life Expectancy** | Expected operational lifespan of vessels (typically 20 years) |
| **LNG** | Liquefied Natural Gas |
| **m³ / cbm** | Cubic meters (volume unit for LNG cargo in vessel capacity) |
| **MEGI** | M-type Electronically Controlled Gas Injection (vessel propulsion) |
| **Net Balance** | total_active_ships - ships_demand |
| **Round-Trip Time** | laden_days + nonladen_days |
| **Ships Demand** | Number of vessels needed to meet trade volume for period |
| **Shipping Region** | Geographic grouping (e.g., 'US Gulf', 'North Europe') |
| **Supply View** | Export-centric perspective measuring vessels at loading ports |
| **Utilization Rate** | Fraction of vessel capacity filled with cargo (0.0-1.0) |
| **Vessel-Period** | Unit for ships demand (e.g., 12.5 vessel-months = 12 full + 1 half-utilized) |
| **Volume-Weighted** | Metric averaged with cargo volume as weight, not trade count |
| **Window** | 365-day rolling period used to calculate future projection patterns |
| **WoodMac** | Wood Mackenzie (LNG forecast data provider) |

---

## 15. References & Data Sources

### 15.1 External Data Sources

1. **Kpler Trades** (`at_lng.kpler_trades`)
   - Provider: Kpler SAS
   - Update Frequency: Daily
   - Coverage: Global LNG trade flows
   - Key Fields: vessel_name, start, end, cargo volumes, status

2. **Wood Mackenzie Forecasts** (`at_lng.woodmac_gas_imports_exports_monthly__mmtpa`)
   - Provider: Wood Mackenzie
   - Update Frequency: Monthly (Short Term), Quarterly (Long Term)
   - Coverage: Country-level import/export forecasts through 2050
   - Key Fields: metric_value (MMTPA), direction, measured_at

3. **Kpler Vessel Info** (`at_lng.kpler_vessels_info`)
   - Provider: Kpler SAS
   - Update Frequency: Weekly
   - Coverage: Global LNG vessel fleet characteristics
   - Key Fields: vessel_capacity_cubic_meters, vessel_status, vessel_build_year

4. **SYY Newbuilds** (`at_lng.syy_newbuilds`)
   - Provider: Shipyards & Yards database
   - Update Frequency: Monthly
   - Coverage: LNG vessels under construction
   - Key Fields: Delivery date, Capacity

### 15.2 Mapping Tables

1. **Country-Region Mapping** (`at_lng.mappings_country`)
   - Maps countries to shipping regions
   - Example: France → North Europe

2. **Vessel Type-Capacity Mapping** (`at_lng.mapping_vessel_type_capacity`)
   - Classifies vessels by capacity ranges
   - Example: 165,000-180,000 cbm → Q-Max

---

## 16. Document Maintenance

**Document Owner:** Shipping Analytics Team
**Review Frequency:** Quarterly
**Last Technical Review:** 2025-09-30
**Next Scheduled Review:** 2025-12-31

**Change Log:**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-09-30 | Technical Documentation Team | Initial comprehensive documentation |

---

## Appendix A: Complete Function Call Tree

```
global_shipping_balance()
├── fetch_woodmac_data()
├── get_max_historical_date()
├── calculate_shipping_metrics()
│   ├── kpler_analysis()
│   │   ├── SQL: supply_trades CTE
│   │   ├── Ballast leg reconstruction
│   │   ├── Volume-weighted aggregation (laden)
│   │   └── Capacity-weighted aggregation (non-laden)
├── calculate_historical_metrics()
│   └── kpler_analysis()
├── process_shipping_demand()
│   ├── process_historical_demand()
│   │   └── Volume-weighted aggregation by Date
│   └── process_future_demand()
│       └── Volume-weighted aggregation by Date
├── analyze_vessel_fleet()
│   ├── SQL: Kpler vessels
│   ├── SQL: SYY newbuilds
│   ├── distribute_retirement_dates()
│   └── calculate_monthly_vessel_stats()
├── calculate_final_balance()
│   ├── get_days_in_period()
│   └── Ships demand calculation (historical & future)
└── Return: df_global_shipping_balance
```

---

**END OF TECHNICAL DOCUMENTATION**

*This document provides a complete technical specification of the Global LNG Shipping Balance calculation system. For questions or clarifications, contact the Shipping Analytics Team.*