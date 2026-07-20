from io import BytesIO

import pandas as pd

from pages import capacity


def _scenario_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "scenario_row_key": "saved-event-1",
                "terminal_key": "terminal-1",
                "train_key": "train-1",
                "country_name": "Country A",
                "plant_name": "Plant A",
                "train_label": "Train 1 (GL1Z)",
                "base_provider": "woodmac",
                "base_first_date": "2028-01-01",
                "base_capacity_mtpa": 4.0,
                "scenario_first_date": "2029-01-01",
                "scenario_capacity_mtpa": 4.0,
                "scenario_note": "original note",
                "display_sort_plant": "Plant A",
                "display_sort_train": 1,
            },
            {
                "scenario_row_key": "saved-event-2",
                "terminal_key": "terminal-1",
                "train_key": "train-1",
                "country_name": "Country A",
                "plant_name": "Plant A",
                "train_label": "1",
                "base_provider": "internal_scenario",
                "base_first_date": "2030-01-01",
                "base_capacity_mtpa": 1.0,
                "scenario_first_date": "2030-01-01",
                "scenario_capacity_mtpa": 1.0,
                "scenario_note": "",
                "display_sort_plant": "Plant A",
                "display_sort_train": 1,
            },
        ]
    )


def _round_tripped_workbook(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    export_df = pd.DataFrame(
        {
            capacity.TRAIN_TIMELINE_IMPORT_KEY_COLUMN: rows["scenario_row_key"],
            "Country": rows["country_name"],
            "Plant": rows["plant_name"],
            "Train": rows["train_label"],
            "Scenario First Date": rows["scenario_first_date"],
            "Scenario Capacity": rows["scenario_capacity_mtpa"],
            "Scenario Note": rows["scenario_note"],
        }
    )
    metadata_df = capacity._build_train_timeline_upload_metadata_df(
        export_df,
        selected_scenario_id=99,
        scenario_name="Upload Test",
        original_name_visibility="hide",
        current_rows_df=rows,
    )
    workbook = capacity._export_train_timeline_workbook_bytes(
        export_df,
        metadata_df,
    )
    excel = pd.ExcelFile(BytesIO(workbook))
    return (
        pd.read_excel(excel, sheet_name=capacity.TRAIN_TIMELINE_SHEET_NAME, dtype=object),
        pd.read_excel(
            excel,
            sheet_name=capacity.TRAIN_TIMELINE_IMPORT_META_SHEET_NAME,
            dtype=object,
        ),
    )


def test_timeline_upload_restores_deleted_saved_row_with_canonical_identity():
    original_rows = _scenario_rows()
    main_df, metadata_df = _round_tripped_workbook(original_rows)

    unchanged, unchanged_summary = capacity._build_train_timeline_upload_rows_df(
        main_df,
        metadata_df,
        selected_scenario_id=99,
        current_rows_df=original_rows,
    )
    assert unchanged_summary == {"updated": 0, "added": 0, "deleted": 0}
    assert len(unchanged) == 2

    current_rows = original_rows[original_rows["scenario_row_key"] != "saved-event-1"]
    restored, restored_summary = capacity._build_train_timeline_upload_rows_df(
        main_df,
        metadata_df,
        selected_scenario_id=99,
        current_rows_df=current_rows,
    )

    assert restored_summary == {"updated": 0, "added": 1, "deleted": 0}
    restored_row = restored.set_index("scenario_row_key").loc["saved-event-1"]
    assert restored_row["train_key"] == "train-1"
    assert restored_row["terminal_key"] == "terminal-1"
    assert restored_row["base_provider"] == "woodmac"
    assert restored_row["base_capacity_mtpa"] == 4.0
    assert restored_row["scenario_capacity_mtpa"] == 4.0
