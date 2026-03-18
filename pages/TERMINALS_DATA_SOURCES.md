# Data Sources and Connections in the Terminals Page

This document explains the data sources, database tables, and data flow architecture used in the Terminals page of the LNG Shipping dashboard.

## Primary Data Sources

### 1. Wood Mackenzie (Woodmac) Data - Baseline Source

**Tables Used:**
- `woodmac_lng_plant_summary` - Plant metadata (plant name, country)
- `woodmac_lng_plant_train` - Train metadata (train IDs, start dates)
- `woodmac_lng_plant_monthly_capacity_nominal_mta` - Monthly capacity data (MTPA)
- `woodmac_lng_plant_train_monthly_output_mta` - Monthly production volumes
- `woodmac_lng_plant_train_annual_output_mta` - Annual production volumes

**Purpose:** Provides the baseline/reference data for all LNG terminal capacity and production forecasts globally.

### 2. Internal Adjustments - Scenario-Based Modifications

**Table Used:**
- `fundamentals_terminals_output_adjustments` - Contains scenario-specific volume adjustments

**Purpose:** Allows users to create custom scenarios (like "best_view", "test_1") that modify the Woodmac baseline with internal forecasts/assumptions.

## Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                            │
│  - Scenario Selector (base_view, best_view, custom scenarios)  │
│  - Unit Selector (MTPA vs Mcm/d)                               │
│  - Year Range Sliders                                          │
│  - Country Filters                                             │
│  - Breakdown Level (Country/Project/Train)                     │
└───────────────────┬─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DATA RETRIEVAL LAYER                         │
│                                                                 │
│  Main Functions:                                               │
│  ├─ fetch_train_data() ────────► Train capacity & start dates │
│  └─ fetch_volume_data() ───────► Monthly/annual volumes       │
└───────────────────┬─────────────────────────────────────────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │  SCENARIO ROUTING    │
         └──────────┬───────────┘
                    │
        ┌───────────┴──────────┐
        │                      │
        ▼                      ▼
┌──────────────────┐  ┌──────────────────────────┐
│   BASE VIEW      │  │   SCENARIO VIEW          │
│ (Woodmac Only)   │  │ (Woodmac + Adjustments) │
└────────┬─────────┘  └──────────┬───────────────┘
         │                       │
         │                       │
         ▼                       ▼
┌─────────────────────────────────────────────┐
│        DATABASE QUERIES (PostgreSQL)        │
│                                             │
│  WOODMAC BASELINE:                         │
│  ┌──────────────────────────────────────┐  │
│  │ 1. woodmac_lng_plant_summary        │  │
│  │    ↓ (JOIN by id_plant)             │  │
│  │ 2. woodmac_lng_plant_train          │  │
│  │    ↓ (JOIN by id_plant, id_lng_train)│ │
│  │ 3. Monthly Capacity (nominal_mta)   │  │
│  │    ↓ (JOIN by id_plant, id_lng_train)│ │
│  │ 4. Monthly Output (output_mta)      │  │
│  │    ↓ (UNION with Annual where gaps) │  │
│  │ 5. Annual Output (annual_output)    │  │
│  └──────────────────────────────────────┘  │
│                                             │
│  SCENARIO ADJUSTMENTS (if not base_view):  │
│  ┌──────────────────────────────────────┐  │
│  │ fundamentals_terminals_output_       │  │
│  │ adjustments                          │  │
│  │   ↓ (LEFT JOIN by train, year, month)│ │
│  │ COALESCE(adjusted, baseline)         │  │
│  └──────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│         DATA PROCESSING LAYER               │
│                                             │
│  - Filter by year range                    │
│  - Filter by countries                     │
│  - Filter new capacity only (≥ today)      │
│  - Convert units (MTPA ↔ Mcm/d)           │
│  - Aggregate by breakdown level:           │
│    * Country: GROUP BY country_name        │
│    * Project: GROUP BY plant_name          │
│    * Train: Individual train level         │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│         VISUALIZATION LAYER                 │
│                                             │
│  Chart 1: Timeline (Gantt-style bars)      │
│  - Shows train start dates                 │
│  - Capacity as bar height                  │
│  - Arrows show adjustments vs baseline     │
│                                             │
│  Chart 2: Volume Area Chart                │
│  - Stacked area showing cumulative output  │
│  - Color-coded by country/project/train    │
│  - Time series of monthly volumes          │
└─────────────────────────────────────────────┘
```

## Key Data Relationships

### Data Hierarchy

```
Country
  └─ Plant/Terminal (id_plant)
       └─ Train (id_lng_train)
            ├─ Capacity (MTPA)
            ├─ Start Date
            └─ Monthly/Annual Output
```

### Data Integration Logic

1. **Monthly data takes precedence** over annual data (higher granularity)
2. **Annual data fills gaps** beyond monthly coverage by expanding to 12 monthly values
3. **Adjustments override baseline** via `COALESCE(adjusted_output, baseline_output)`
4. **Latest data wins** using `DISTINCT ON` with `ORDER BY upload_timestamp_utc DESC`

### Scenario Mechanism

- **base_view**: Pure Woodmac data, no adjustments applied
- **Other scenarios**: Woodmac baseline + custom adjustments from `fundamentals_terminals_output_adjustments`
- Adjustments are optional - trains without adjustments use baseline values
- Start dates can be modified in scenarios (shown with arrows in timeline chart)

## Data Processing Steps

1. **Query Execution:** Complex CTEs (Common Table Expressions) retrieve and deduplicate data
2. **Filtering:** Apply year ranges, country selections, and new capacity filters
3. **Unit Conversion:** Convert MTPA to Mcm/d using formula: `MTPA * 1.36 / 365 * 1000`
4. **Aggregation:** Sum/group by selected breakdown level (country/project/train)
5. **Visualization:** Create Plotly charts with color-coding and interactive features

## Key Functions

### fetch_train_data(scenario='base_view')

Fetches train-level data including:
- Plant name and country
- Train ID and start date
- Capacity (derived from max monthly/annual capacity)
- Data source indicator (baseline vs adjusted)

**Parameters:**
- `scenario`: Scenario name ('base_view', 'best_view', etc.)

**Returns:** DataFrame with train information including derived start dates

### fetch_volume_data(start_year, end_year, breakdown, new_capacity_only, selected_countries, scenario)

Fetches monthly output volume data with different breakdown levels.

**Parameters:**
- `start_year`: Starting year for data
- `end_year`: Ending year for data
- `breakdown`: 'country', 'project', or 'train'
- `new_capacity_only`: If True, filter to show only new capacity from current month onwards
- `selected_countries`: List of countries to filter by
- `scenario`: Scenario name for adjustments

**Returns:** DataFrame with aggregated volumes by selected breakdown level

### create_timeline_figure(selected_unit, scenario, year_range)

Creates the Gantt-style timeline chart showing train start dates and capacity additions.

**Features:**
- Bar height represents capacity
- Color-coded by country
- Arrows show adjustments vs Woodmac baseline
- Train count indicators for multiple trains in same month

### create_volume_area_chart(selected_unit, breakdown, new_capacity_only, selected_countries, scenario, start_year, end_year)

Creates stacked area chart showing cumulative monthly output.

**Features:**
- Stacked areas by country/project/train
- Time series from start_year to end_year
- Color-coded for visual distinction
- Interactive tooltips with detailed information

## External Dependencies

- **scenario_utils.py** (`fundamentals/terminals/scenario_utils.py`): Provides `get_available_scenarios()` function to populate scenario dropdown
- **config.ini**: Database connection string and schema name
- **SQLAlchemy Engine**: Database connectivity layer

## Database Schema

All tables are in the `at_lng` schema in the PostgreSQL/Neon database. Connection details are stored in `config.ini`.

## Notes

- The page uses complex SQL queries with CTEs for performance optimization
- Monthly data is prioritized over annual data for accuracy
- The scenario system allows flexible forecasting without modifying baseline data
- All visualizations support both MTPA and Mcm/d units
- Export to Excel functionality preserves all filtering and aggregation settings
