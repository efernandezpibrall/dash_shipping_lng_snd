import sys
from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "dash_shipping_lng_snd"))

from pages import capacity  # noqa: E402


def _build_woodmac_raw_df(records: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        records,
        columns=["month", "country_name", "total_mmtpa"],
    )


def _build_ea_raw_df(records: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "month": month,
                "country_name": country_name,
                "capacity_mtpa": capacity_mtpa,
                "status": "active",
            }
            for month, country_name, capacity_mtpa in records
        ]
    )


def _build_scenario_rows(records: list[tuple[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "scenario_row_key": f"row-{index}",
                "country_name": "Qatar",
                "plant_name": "North Field",
                "train_label": "1",
                "scenario_first_date": pd.Timestamp(first_date),
                "scenario_capacity_mtpa": capacity_mtpa,
            }
            for index, (first_date, capacity_mtpa) in enumerate(records, start=1)
        ]
    )


def _build_woodmac_discrepancy_change_df(
    records: list[tuple[str, str, str, object, float]]
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Effective Date": effective_date,
                "Country": country,
                "Plant": plant,
                "Train": train,
                "Delta MTPA": capacity_mtpa,
            }
            for effective_date, country, plant, train, capacity_mtpa in records
        ]
    )


def _build_ea_discrepancy_change_df(
    records: list[tuple[str, str, str, object, float]]
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Effective Date": effective_date,
                "Country": country,
                "Plant": project,
                "Train": train,
                "EA Net Delta (MTPA)": capacity_mtpa,
            }
            for effective_date, country, project, train, capacity_mtpa in records
        ]
    )


def _build_internal_discrepancy_change_df(
    records: list[tuple[str, str, str, object, float]]
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Effective Date": effective_date,
                "Country": country,
                "Plant": plant,
                "Train": train,
                capacity.INTERNAL_SCENARIO_NET_COLUMN: capacity_mtpa,
            }
            for effective_date, country, plant, train, capacity_mtpa in records
        ]
    )


def _collect_component_ids(component) -> list[str]:
    if component is None:
        return []
    if isinstance(component, (list, tuple)):
        ids: list[str] = []
        for item in component:
            ids.extend(_collect_component_ids(item))
        return ids
    if isinstance(component, (str, int, float, bool)):
        return []
    if not hasattr(component, "to_plotly_json"):
        return []

    props = component.to_plotly_json().get("props", {})
    ids = [props["id"]] if props.get("id") else []
    ids.extend(_collect_component_ids(props.get("children")))
    return ids


def test_build_yearly_capacity_comparison_df_uses_december_snapshots():
    woodmac_raw_df = _build_woodmac_raw_df(
        [
            ("2024-01-01", "Qatar", 1.0),
            ("2024-12-01", "Qatar", 12.0),
            ("2025-01-01", "Qatar", 13.0),
            ("2025-12-01", "Qatar", 24.0),
        ]
    )
    ea_raw_df = _build_ea_raw_df(
        [
            ("2024-01-01", "Qatar", 80.0),
            ("2025-12-01", "Qatar", 0.0),
        ]
    )
    scenario_rows_df = _build_scenario_rows(
        [
            ("2024-01-01", 5.0),
            ("2024-07-01", 7.0),
            ("2025-12-01", 8.0),
        ]
    )

    result = capacity._build_yearly_capacity_comparison_df(
        woodmac_raw_df,
        ea_raw_df,
        scenario_rows_df,
        None,
        None,
    )

    assert result.to_dict("records") == [
        {
            "Year": "2024",
            "Internal Scenario": 12.0,
            "Woodmac": 12.0,
            "Energy Aspects": 80.0,
            "Delta vs Woodmac": 0.0,
            "Delta vs Energy Aspects": -68.0,
        },
        {
            "Year": "2025",
            "Internal Scenario": 20.0,
            "Woodmac": 24.0,
            "Energy Aspects": 80.0,
            "Delta vs Woodmac": -4.0,
            "Delta vs Energy Aspects": -60.0,
        },
    ]


def test_build_yearly_capacity_comparison_df_keeps_only_provider_overlap_years():
    woodmac_raw_df = _build_woodmac_raw_df(
        [
            ("2024-12-01", "Qatar", 10.0),
            ("2025-12-01", "Qatar", 20.0),
            ("2026-12-01", "Qatar", 30.0),
        ]
    )
    ea_raw_df = _build_ea_raw_df(
        [
            ("2025-12-01", "Qatar", 50.0),
            ("2027-12-01", "Qatar", 0.0),
        ]
    )
    scenario_rows_df = _build_scenario_rows(
        [
            ("2025-12-01", 3.0),
            ("2026-12-01", 4.0),
        ]
    )

    result = capacity._build_yearly_capacity_comparison_df(
        woodmac_raw_df,
        ea_raw_df,
        scenario_rows_df,
        None,
        None,
    )

    assert result["Year"].tolist() == ["2025", "2026"]


def test_build_yearly_capacity_comparison_df_leaves_internal_and_deltas_blank_for_missing_years():
    woodmac_raw_df = _build_woodmac_raw_df(
        [
            ("2024-12-01", "Qatar", 10.0),
            ("2025-12-01", "Qatar", 12.0),
        ]
    )
    ea_raw_df = _build_ea_raw_df(
        [
            ("2024-12-01", "Qatar", 30.0),
            ("2025-12-01", "Qatar", 0.0),
        ]
    )
    scenario_rows_df = _build_scenario_rows(
        [
            ("2025-12-01", 4.0),
        ]
    )

    result = capacity._build_yearly_capacity_comparison_df(
        woodmac_raw_df,
        ea_raw_df,
        scenario_rows_df,
        None,
        None,
    )

    assert result["Year"].tolist() == ["2024", "2025"]
    assert pd.isna(result.loc[0, "Internal Scenario"])
    assert pd.isna(result.loc[0, "Delta vs Woodmac"])
    assert pd.isna(result.loc[0, "Delta vs Energy Aspects"])
    assert result.loc[1, "Internal Scenario"] == 4.0
    assert result.loc[1, "Delta vs Woodmac"] == -8.0
    assert result.loc[1, "Delta vs Energy Aspects"] == -26.0


def test_render_yearly_capacity_comparison_section_requires_selected_scenario():
    figure, table = capacity.render_yearly_capacity_comparison_section(
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )

    assert figure.layout.annotations[0].text == capacity.YEARLY_CAPACITY_COMPARISON_EMPTY_MESSAGE
    assert table.children == capacity.YEARLY_CAPACITY_COMPARISON_EMPTY_MESSAGE


def test_render_yearly_capacity_discrepancy_section_requires_selected_scenario():
    (
        woodmac_capacity_table,
        woodmac_timeline_table,
        woodmac_missing_table,
        ea_capacity_table,
        ea_timeline_table,
        ea_missing_table,
    ) = capacity.render_yearly_capacity_discrepancy_section(
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )

    assert capacity.YEARLY_CAPACITY_COMPARISON_EMPTY_MESSAGE in str(woodmac_capacity_table)
    assert capacity.YEARLY_CAPACITY_COMPARISON_EMPTY_MESSAGE in str(woodmac_timeline_table)
    assert capacity.YEARLY_CAPACITY_COMPARISON_EMPTY_MESSAGE in str(woodmac_missing_table)
    assert capacity.YEARLY_CAPACITY_COMPARISON_EMPTY_MESSAGE in str(ea_capacity_table)
    assert capacity.YEARLY_CAPACITY_COMPARISON_EMPTY_MESSAGE in str(ea_timeline_table)
    assert capacity.YEARLY_CAPACITY_COMPARISON_EMPTY_MESSAGE in str(ea_missing_table)


def test_create_yearly_capacity_comparison_chart_starts_at_min_and_uses_50_tick_spacing():
    comparison_df = pd.DataFrame(
        [
            {
                "Year": "2024",
                "Internal Scenario": 120.0,
                "Woodmac": 140.0,
                "Energy Aspects": 180.0,
                "Delta vs Woodmac": -20.0,
                "Delta vs Energy Aspects": -60.0,
            },
            {
                "Year": "2025",
                "Internal Scenario": 170.0,
                "Woodmac": 160.0,
                "Energy Aspects": 210.0,
                "Delta vs Woodmac": 10.0,
                "Delta vs Energy Aspects": -40.0,
            },
        ]
    )

    figure = capacity._create_yearly_capacity_comparison_chart(comparison_df)

    assert figure.layout.yaxis.range[0] == 120.0
    assert figure.layout.yaxis.dtick == 50


def test_build_provider_capacity_discrepancy_df_aggregates_and_ranks_woodmac_rows():
    result = capacity._build_provider_capacity_discrepancy_df(
        "woodmac",
        _build_woodmac_discrepancy_change_df(
            [
                ("2024-01-01", "Qatar", "North Field", 1, 10.0),
                ("2024-03-01", "Qatar", "North Field", 1, 5.0),
                ("2024-02-01", "Qatar", "Alpha Project", "", 9.0),
                ("2024-04-01", "Qatar", "Beta Train", 2, 2.0),
            ]
        ),
        _build_internal_discrepancy_change_df(
            [
                ("2024-01-15", "Qatar", "North Field", 1, 6.0),
                ("2024-04-01", "Qatar", "North Field", 1, 1.0),
                ("2024-04-01", "Qatar", "Beta Train", 2, 2.0),
            ]
        ),
    )

    assert result["Plant"].tolist() == ["Alpha Project", "North Field"]
    assert result["Train"].tolist() == ["", "1"]
    assert result["Woodmac First Date"].tolist() == ["2024-02-01", "2024-01-01"]
    assert result["Woodmac Capacity"].tolist() == [9.0, 15.0]
    assert pd.isna(result.loc[0, "Scenario First Date"])
    assert pd.isna(result.loc[0, "Scenario Capacity"])
    assert result.loc[1, "Scenario First Date"] == "2024-01-15"
    assert result.loc[1, "Scenario Capacity"] == 7.0
    assert result["Abs Capacity Delta"].tolist() == [9.0, 8.0]


def test_build_provider_capacity_discrepancy_df_builds_energy_aspects_project_columns():
    result = capacity._build_provider_capacity_discrepancy_df(
        "energy_aspects",
        _build_ea_discrepancy_change_df(
            [
                ("2024-05-01", "United States", "Golden Pass", 1, 12.5),
            ]
        ),
        _build_internal_discrepancy_change_df(
            [
                ("2024-06-01", "United States", "Golden Pass", 1, 9.5),
            ]
        ),
    )

    assert result.columns.tolist() == [
        "Country",
        "Project",
        "Train",
        "Energy Aspects First Date",
        "Energy Aspects Capacity",
        "Scenario First Date",
        "Scenario Capacity",
        "Abs Capacity Delta",
    ]
    assert result.loc[0, "Project"] == "Golden Pass"
    assert result.loc[0, "Energy Aspects Capacity"] == 12.5
    assert result.loc[0, "Scenario Capacity"] == 9.5
    assert result.loc[0, "Abs Capacity Delta"] == 3.0


def test_build_provider_capacity_discrepancy_df_limits_to_top_10_rows():
    result = capacity._build_provider_capacity_discrepancy_df(
        "woodmac",
        _build_woodmac_discrepancy_change_df(
            [
                ("2024-01-01", "Qatar", f"Plant {index}", index, float(index))
                for index in range(1, 13)
            ]
        ),
        pd.DataFrame(
            columns=[
                "Effective Date",
                "Country",
                "Plant",
                "Train",
                capacity.INTERNAL_SCENARIO_NET_COLUMN,
            ]
        ),
    )

    assert len(result) == 10
    assert result.iloc[0]["Abs Capacity Delta"] == 12.0
    assert result.iloc[-1]["Abs Capacity Delta"] == 3.0


def test_build_provider_capacity_discrepancy_df_from_timeline_grid_keeps_directions_separate():
    grid_df = pd.DataFrame(
        [
            {
                "Country": "Qatar",
                "Plant": "North Field",
                "Train": "1",
                "timeline_direction": "addition",
                "Woodmac First Date": "2024-01-01",
                "Woodmac Capacity Change": 10.0,
                "Scenario First Date": "2024-02-01",
                "Scenario Capacity": 8.0,
            },
            {
                "Country": "Qatar",
                "Plant": "North Field",
                "Train": "1",
                "timeline_direction": "reduction",
                "Woodmac First Date": "2024-03-01",
                "Woodmac Capacity Change": -6.0,
                "Scenario First Date": "2024-04-01",
                "Scenario Capacity": -1.0,
            },
        ]
    )

    result = capacity._build_provider_capacity_discrepancy_df_from_timeline_grid(
        "woodmac",
        grid_df,
    )

    assert len(result) == 2
    assert result["Plant"].tolist() == ["North Field", "North Field"]
    assert result["Train"].tolist() == ["1", "1"]
    assert result["Woodmac First Date"].tolist() == ["2024-03-01", "2024-01-01"]
    assert result["Woodmac Capacity"].tolist() == [-6.0, 10.0]
    assert result["Scenario Capacity"].tolist() == [-1.0, 8.0]
    assert result["Abs Capacity Delta"].tolist() == [5.0, 2.0]


def test_build_provider_capacity_discrepancy_df_from_timeline_grid_excludes_provider_only_rows():
    grid_df = pd.DataFrame(
        [
            {
                "Country": "Qatar",
                "Plant": "North Field",
                "Train": "1",
                "timeline_direction": "addition",
                "Woodmac First Date": "2024-01-01",
                "Woodmac Capacity Change": 10.0,
                "Scenario First Date": None,
                "Scenario Capacity": None,
            },
            {
                "Country": "Qatar",
                "Plant": "Alpha",
                "Train": "2",
                "timeline_direction": "addition",
                "Woodmac First Date": "2024-02-01",
                "Woodmac Capacity Change": 8.0,
                "Scenario First Date": "2024-03-01",
                "Scenario Capacity": 5.0,
            },
        ]
    )

    result = capacity._build_provider_capacity_discrepancy_df_from_timeline_grid(
        "woodmac",
        grid_df,
    )

    assert result["Plant"].tolist() == ["Alpha"]
    assert result["Abs Capacity Delta"].tolist() == [3.0]


def test_build_provider_capacity_discrepancy_df_from_timeline_grid_keeps_all_rows():
    grid_df = pd.DataFrame(
        [
            {
                "Country": "Qatar",
                "Plant": f"Plant {index}",
                "Train": str(index),
                "timeline_direction": "addition",
                "Woodmac First Date": "2024-01-01",
                "Woodmac Capacity Change": float(index),
                "Scenario First Date": "2024-02-01",
                "Scenario Capacity": 0.0,
            }
            for index in range(1, 13)
        ]
    )

    result = capacity._build_provider_capacity_discrepancy_df_from_timeline_grid(
        "woodmac",
        grid_df,
    )

    assert len(result) == 12
    assert result.iloc[0]["Abs Capacity Delta"] == 12.0
    assert result.iloc[-1]["Abs Capacity Delta"] == 1.0


def test_build_provider_timeline_discrepancy_df_from_timeline_grid_uses_month_gaps():
    grid_df = pd.DataFrame(
        [
            {
                "Country": "Qatar",
                "Plant": "North Field",
                "Train": "1",
                "timeline_direction": "addition",
                "Woodmac First Date": "2024-01-15",
                "Scenario First Date": "2024-03-31",
            },
            {
                "Country": "Qatar",
                "Plant": "Alpha",
                "Train": "2",
                "timeline_direction": "addition",
                "Woodmac First Date": "2024-01-01",
                "Scenario First Date": "2024-01-31",
            },
            {
                "Country": "Qatar",
                "Plant": "Beta",
                "Train": "3",
                "timeline_direction": "addition",
                "Woodmac First Date": "2024-02-01",
                "Scenario First Date": "2024-03-01",
            },
            {
                "Country": "Qatar",
                "Plant": "Gamma",
                "Train": "4",
                "timeline_direction": "addition",
                "Woodmac First Date": "2024-04-01",
                "Scenario First Date": None,
            },
        ]
    )

    result = capacity._build_provider_timeline_discrepancy_df_from_timeline_grid(
        "woodmac",
        grid_df,
    )

    assert result["Plant"].tolist() == ["North Field", "Beta"]
    assert result["Abs Timeline Delta (Months)"].tolist() == [2, 1]
    assert result["Woodmac First Date"].tolist() == ["2024-01-15", "2024-02-01"]
    assert result["Scenario First Date"].tolist() == ["2024-03-31", "2024-03-01"]


def test_build_provider_timeline_discrepancy_df_from_timeline_grid_keeps_directions_separate():
    grid_df = pd.DataFrame(
        [
            {
                "Country": "Qatar",
                "Plant": "North Field",
                "Train": "1",
                "timeline_direction": "addition",
                "Woodmac First Date": "2024-01-01",
                "Scenario First Date": "2024-02-01",
            },
            {
                "Country": "Qatar",
                "Plant": "North Field",
                "Train": "1",
                "timeline_direction": "reduction",
                "Woodmac First Date": "2024-03-01",
                "Scenario First Date": "2024-08-01",
            },
        ]
    )

    result = capacity._build_provider_timeline_discrepancy_df_from_timeline_grid(
        "woodmac",
        grid_df,
    )

    assert result["Plant"].tolist() == ["North Field", "North Field"]
    assert result["Train"].tolist() == ["1", "1"]
    assert result["Abs Timeline Delta (Months)"].tolist() == [5, 1]
    assert result["Woodmac First Date"].tolist() == ["2024-03-01", "2024-01-01"]


def test_build_provider_timeline_discrepancy_df_from_timeline_grid_keeps_all_rows():
    grid_df = pd.DataFrame(
        [
            {
                "Country": "Qatar",
                "Plant": f"Plant {index}",
                "Train": str(index),
                "timeline_direction": "addition",
                "Woodmac First Date": "2024-01-01",
                "Scenario First Date": f"2025-{index:02d}-01",
            }
            for index in range(1, 13)
        ]
    )

    result = capacity._build_provider_timeline_discrepancy_df_from_timeline_grid(
        "woodmac",
        grid_df,
    )

    assert len(result) == 12
    assert result.iloc[0]["Abs Timeline Delta (Months)"] == 23
    assert result.iloc[-1]["Abs Timeline Delta (Months)"] == 12


def test_build_provider_missing_internal_scenario_df_from_timeline_grid_captures_provider_only_rows():
    grid_df = pd.DataFrame(
        [
            {
                "Country": "Qatar",
                "Plant": "North Field",
                "Train": "1",
                "timeline_direction": "addition",
                "Woodmac First Date": "2024-01-01",
                "Scenario First Date": None,
                "Woodmac Capacity Change": 9.0,
                "Scenario Capacity": None,
            },
            {
                "Country": "Qatar",
                "Plant": "Alpha",
                "Train": "2",
                "timeline_direction": "addition",
                "Woodmac First Date": None,
                "Scenario First Date": "2024-04-01",
                "Woodmac Capacity Change": None,
                "Scenario Capacity": 5.0,
            },
            {
                "Country": "Qatar",
                "Plant": "Beta",
                "Train": "3",
                "timeline_direction": "addition",
                "Woodmac First Date": None,
                "Scenario First Date": None,
                "Woodmac Capacity Change": 4.0,
                "Scenario Capacity": None,
            },
            {
                "Country": "Qatar",
                "Plant": "Gamma",
                "Train": "4",
                "timeline_direction": "addition",
                "Woodmac First Date": "2024-01-01",
                "Scenario First Date": "2024-03-01",
                "Woodmac Capacity Change": 7.0,
                "Scenario Capacity": 6.0,
            },
        ]
    )

    result = capacity._build_provider_missing_internal_scenario_df_from_timeline_grid(
        "woodmac",
        grid_df,
    )

    assert result["Plant"].tolist() == ["Beta", "North Field"]
    assert result["Woodmac Capacity"].tolist() == [4.0, 9.0]


def test_create_yearly_sections_split_ids():
    comparison_section = capacity._create_yearly_capacity_comparison_section(
        "Yearly Capacity Comparison",
        "subtitle",
        "capacity-page-yearly-capacity-comparison-chart",
        "capacity-page-yearly-capacity-comparison-table-container",
    )
    discrepancy_section = capacity._create_yearly_provider_discrepancy_section(
        "Provider Discrepancies",
        "subtitle",
    )

    comparison_ids = _collect_component_ids(comparison_section)
    discrepancy_ids = _collect_component_ids(discrepancy_section)

    assert "capacity-page-yearly-capacity-comparison-chart" in comparison_ids
    assert "capacity-page-yearly-capacity-comparison-table-container" in comparison_ids
    assert "capacity-page-yearly-woodmac-capacity-discrepancy-table-container" not in comparison_ids
    assert discrepancy_ids.index(
        "capacity-page-yearly-woodmac-capacity-discrepancy-table-container"
    ) < discrepancy_ids.index("capacity-page-yearly-woodmac-timeline-discrepancy-table-container")
    assert discrepancy_ids.index(
        "capacity-page-yearly-woodmac-timeline-discrepancy-table-container"
    ) < discrepancy_ids.index("capacity-page-yearly-woodmac-missing-internal-table-container")
    assert discrepancy_ids.index(
        "capacity-page-yearly-woodmac-missing-internal-table-container"
    ) < discrepancy_ids.index("capacity-page-yearly-ea-capacity-discrepancy-table-container")
    assert discrepancy_ids.index(
        "capacity-page-yearly-ea-capacity-discrepancy-table-container"
    ) < discrepancy_ids.index("capacity-page-yearly-ea-timeline-discrepancy-table-container")
    assert discrepancy_ids.index(
        "capacity-page-yearly-ea-timeline-discrepancy-table-container"
    ) < discrepancy_ids.index("capacity-page-yearly-ea-missing-internal-table-container")


def test_create_yearly_capacity_comparison_table_uses_yearly_column_widths():
    comparison_df = pd.DataFrame(
        [
            {
                "Year": "2024",
                "Internal Scenario": 123.4,
                "Woodmac": 140.0,
                "Energy Aspects": 180.0,
                "Delta vs Woodmac": -16.6,
                "Delta vs Energy Aspects": -56.6,
            }
        ]
    )

    table = capacity._create_yearly_capacity_comparison_table(
        "capacity-page-yearly-capacity-comparison-table",
        comparison_df,
    )
    props = table.to_plotly_json()["props"]

    assert props["fill_width"] is True
    assert "fixed_columns" not in props
    assert props["style_cell_conditional"] == capacity._build_yearly_capacity_comparison_column_styles()


def test_create_provider_capacity_discrepancy_table_uses_yearly_column_widths():
    discrepancy_df = pd.DataFrame(
        [
            {
                "Country": "United States",
                "Plant": "Woodside Louisiana LNG",
                "Train": 4,
                "Woodmac First Date": "2031-01-01",
                "Woodmac Capacity": 5.5,
                "Scenario First Date": "2031-04-01",
                "Scenario Capacity": 5.5,
                "Abs Capacity Delta": 5.5,
            }
        ]
    )

    table = capacity._create_provider_capacity_discrepancy_table(
        "capacity-page-yearly-capacity-woodmac-discrepancy-table",
        discrepancy_df,
        "woodmac",
    )
    props = table.to_plotly_json()["props"]

    assert props["fill_width"] is True
    assert props["style_cell_conditional"] == capacity._build_yearly_discrepancy_column_styles("woodmac")
    assert props["style_table"]["maxHeight"] == capacity.YEARLY_PROVIDER_DISCREPANCY_TABLE_MAX_HEIGHT


def test_create_provider_timeline_discrepancy_table_uses_yearly_timeline_widths():
    discrepancy_df = pd.DataFrame(
        [
            {
                "Country": "United States",
                "Plant": "Woodside Louisiana LNG",
                "Train": 4,
                "Woodmac First Date": "2031-01-01",
                "Scenario First Date": "2031-04-01",
                "Abs Timeline Delta (Months)": 3,
            }
        ]
    )

    table = capacity._create_provider_timeline_discrepancy_table(
        "capacity-page-yearly-woodmac-timeline-discrepancy-table",
        discrepancy_df,
        "woodmac",
    )
    props = table.to_plotly_json()["props"]

    assert props["fill_width"] is True
    assert props["style_cell_conditional"] == capacity._build_yearly_timeline_discrepancy_column_styles(
        "woodmac"
    )
    assert props["style_table"]["maxHeight"] == capacity.YEARLY_PROVIDER_DISCREPANCY_TABLE_MAX_HEIGHT


def test_create_provider_missing_internal_scenario_table_uses_yearly_widths():
    missing_df = pd.DataFrame(
        [
            {
                "Country": "United States",
                "Plant": "Woodside Louisiana LNG",
                "Train": 4,
                "Woodmac First Date": "2031-01-01",
                "Woodmac Capacity": 5.5,
            }
        ]
    )

    table = capacity._create_provider_missing_internal_scenario_table(
        "capacity-page-yearly-woodmac-missing-internal-table",
        missing_df,
        "woodmac",
    )
    props = table.to_plotly_json()["props"]

    assert props["fill_width"] is True
    assert props["style_cell_conditional"] == capacity._build_yearly_missing_internal_column_styles(
        "woodmac"
    )
    assert props["style_table"]["maxHeight"] == capacity.YEARLY_PROVIDER_DISCREPANCY_TABLE_MAX_HEIGHT


def test_render_yearly_capacity_comparison_section_populates_chart_and_table(monkeypatch):
    comparison_df = pd.DataFrame(
        [
            {
                "Year": "2024",
                "Internal Scenario": 120.0,
                "Woodmac": 140.0,
                "Energy Aspects": 180.0,
                "Delta vs Woodmac": -20.0,
                "Delta vs Energy Aspects": -60.0,
            }
        ]
    )

    monkeypatch.setattr(capacity, "_resolve_active_capacity_scenario_rows", lambda *args: pd.DataFrame())
    monkeypatch.setattr(
        capacity,
        "_build_yearly_capacity_comparison_df",
        lambda *args: comparison_df,
    )

    figure, table = capacity.render_yearly_capacity_comparison_section(
        capacity._serialize_dataframe(pd.DataFrame()),
        capacity._serialize_dataframe(pd.DataFrame()),
        7,
        "2024-01-01",
        "2024-12-31",
        None,
        None,
        None,
        None,
    )

    assert len(figure.data) == 3
    assert table.to_plotly_json()["props"]["id"] == "capacity-page-yearly-capacity-comparison-table"


def test_render_yearly_capacity_discrepancy_section_populates_provider_outputs(monkeypatch):
    payloads = {
        "woodmac": {
            "capacity_df": pd.DataFrame(
                [
                    {
                        "Country": "Qatar",
                        "Plant": "North Field",
                        "Train": "1",
                        "Woodmac First Date": "2024-01-01",
                        "Woodmac Capacity": 10.0,
                        "Scenario First Date": "2024-02-01",
                        "Scenario Capacity": 8.0,
                        "Abs Capacity Delta": 2.0,
                    }
                ]
            ),
            "timeline_df": pd.DataFrame(
                [
                    {
                        "Country": "Qatar",
                        "Plant": "North Field",
                        "Train": "1",
                        "Woodmac First Date": "2024-01-01",
                        "Scenario First Date": "2024-03-01",
                        "Abs Timeline Delta (Months)": 2,
                    }
                ]
            ),
            "missing_df": pd.DataFrame(
                [
                    {
                        "Country": "Qatar",
                        "Plant": "Alpha",
                        "Train": "2",
                        "Woodmac First Date": "2024-04-01",
                        "Woodmac Capacity": 4.0,
                    }
                ]
            ),
        },
        "energy_aspects": {
            "capacity_df": pd.DataFrame(
                [
                    {
                        "Country": "United States",
                        "Project": "Golden Pass",
                        "Train": "1",
                        "Energy Aspects First Date": "2024-05-01",
                        "Energy Aspects Capacity": 12.5,
                        "Scenario First Date": "2024-06-01",
                        "Scenario Capacity": 9.5,
                        "Abs Capacity Delta": 3.0,
                    }
                ]
            ),
            "timeline_df": pd.DataFrame(
                [
                    {
                        "Country": "United States",
                        "Project": "Golden Pass",
                        "Train": "1",
                        "Energy Aspects First Date": "2024-05-01",
                        "Scenario First Date": "2024-08-01",
                        "Abs Timeline Delta (Months)": 3,
                    }
                ]
            ),
            "missing_df": pd.DataFrame(
                [
                    {
                        "Country": "United States",
                        "Project": "Cameron LNG",
                        "Train": "2",
                        "Energy Aspects First Date": "2024-07-01",
                        "Energy Aspects Capacity": 6.0,
                    }
                ]
            ),
        },
    }

    monkeypatch.setattr(capacity, "_resolve_active_capacity_scenario_rows", lambda *args: pd.DataFrame())
    monkeypatch.setattr(
        capacity,
        "_build_yearly_provider_discrepancy_payloads",
        lambda *args: payloads,
    )

    outputs = capacity.render_yearly_capacity_discrepancy_section(
        capacity._serialize_dataframe(pd.DataFrame()),
        capacity._serialize_dataframe(pd.DataFrame()),
        7,
        ["Qatar"],
        "exclude",
        "2024-01-01",
        "2024-12-31",
        None,
        None,
        None,
        None,
    )

    assert outputs[0].to_plotly_json()["props"]["id"] == (
        "capacity-page-yearly-woodmac-capacity-discrepancy-table"
    )
    assert outputs[1].to_plotly_json()["props"]["id"] == (
        "capacity-page-yearly-woodmac-timeline-discrepancy-table"
    )
    assert outputs[2].to_plotly_json()["props"]["id"] == (
        "capacity-page-yearly-woodmac-missing-internal-table"
    )
    assert outputs[3].to_plotly_json()["props"]["id"] == (
        "capacity-page-yearly-ea-capacity-discrepancy-table"
    )
    assert outputs[4].to_plotly_json()["props"]["id"] == (
        "capacity-page-yearly-ea-timeline-discrepancy-table"
    )
    assert outputs[5].to_plotly_json()["props"]["id"] == (
        "capacity-page-yearly-ea-missing-internal-table"
    )


def test_sync_capacity_scenario_dropdown_value_applies_explicit_target_selection():
    assert capacity.sync_capacity_scenario_dropdown_value(
        [{"scenario_id": 7, "scenario_name": "Base Case"}],
        7,
        None,
    ) == 7


def test_sync_capacity_scenario_dropdown_value_defaults_to_base_case():
    assert capacity.sync_capacity_scenario_dropdown_value(
        [{"scenario_id": 7, "scenario_name": "Base Case"}],
        None,
        None,
    ) == 7


def test_sync_capacity_scenario_dropdown_value_keeps_valid_current_value_without_target():
    assert (
        capacity.sync_capacity_scenario_dropdown_value(
            [{"scenario_id": 7, "scenario_name": "Base Case"}],
            None,
            7,
        )
        is capacity.no_update
    )


def test_sync_capacity_scenario_dropdown_value_clears_invalid_current_value():
    assert capacity.sync_capacity_scenario_dropdown_value(
        [{"scenario_id": 8, "scenario_name": "Base Case"}],
        None,
        7,
    ) is None
