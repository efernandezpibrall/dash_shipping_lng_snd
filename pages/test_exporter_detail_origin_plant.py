import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "dash_shipping_lng_snd"))

from pages import exporter_detail  # noqa: E402


def test_prepare_origin_plant_summary_scope_uses_selected_destination_level():
    scoped_df = pd.DataFrame(
        {
            "start_date": pd.to_datetime(["2026-04-14", "2026-04-17"]),
            "cargo_mcm": [90.0, 125.0],
            "origin_zone": ["Ras Laffan", "Ras Laffan"],
            "destination_country": ["Kuwait", "Pakistan"],
            "destination_continent": ["Asia", "Asia"],
            "destination_shipping_region": ["Middle East", "Indian Ocean"],
            "destination_basin": ["Indian Ocean Basin", "Indian Ocean Basin"],
            "destination_subcontinent": ["Western Asia", "South Asia"],
            "destination_classification_level1": ["Middle East", "South Asia"],
            "destination_classification": ["Middle East", "South Asia"],
        }
    )

    result = exporter_detail._prepare_origin_plant_summary_scope_df(
        scoped_df,
        "destination_shipping_region",
    )

    assert result[["continent", "country", "cargo_mcm"]].to_dict("records") == [
        {"continent": "Ras Laffan", "country": "Middle East", "cargo_mcm": 90.0},
        {"continent": "Ras Laffan", "country": "Indian Ocean", "cargo_mcm": 125.0},
    ]


def test_prepare_origin_plant_summary_display_expands_destinations_under_zone():
    summary_df = pd.DataFrame(
        {
            "continent": ["Ras Laffan", "Ras Laffan", "Bonny"],
            "country": ["Kuwait", "Pakistan", "Spain"],
            "30D": [10.0, 5.0, 2.0],
            "W16'26": [20.0, 0.0, 3.0],
            "7D": [7.0, 0.0, 0.0],
        }
    )

    result = exporter_detail._prepare_origin_plant_summary_display_df(
        summary_df,
        expanded_zones=["Ras Laffan"],
        rolling_col="30D",
    )

    assert result.to_dict("records") == [
        {"Zone": "▼ Ras Laffan", "Destination": "Total", "30D": 15.0, "W16'26": 20.0, "7D": 7.0},
        {"Zone": "", "Destination": "    Kuwait", "30D": 10.0, "W16'26": 20.0, "7D": 7.0},
        {"Zone": "", "Destination": "    Pakistan", "30D": 5.0, "W16'26": 0.0, "7D": 0.0},
        {"Zone": "▶ Bonny", "Destination": "Total", "30D": 2.0, "W16'26": 3.0, "7D": 0.0},
        {"Zone": "GRAND TOTAL", "Destination": "", "30D": 17.0, "W16'26": 23.0, "7D": 7.0},
    ]
