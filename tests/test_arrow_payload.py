import copy

import numpy as np
import pandas as pd
import pytest

from utils import arrow_payload
from utils import dashboard_snapshot_cache as snapshots


def test_dataframe_arrow_round_trip_preserves_exact_pandas_contract():
    index = pd.Index([30, 10, 20], name="row_id")
    frame = pd.DataFrame(
        {
            "category": pd.Categorical(
                ["loaded", "waiting", None],
                categories=["waiting", "loaded", "idle"],
                ordered=True,
            ),
            "nullable_int": pd.array([3, pd.NA, 2], dtype="Int64"),
            "nullable_bool": pd.array([True, pd.NA, False], dtype="boolean"),
            "quantity": np.array([3.5, np.nan, 2.25], dtype="float64"),
            "date": pd.to_datetime(
                ["2026-03-01", None, "2026-02-01"],
                utc=True,
            ),
            "label": pd.Series(["C", None, "B"], index=index, dtype="string"),
        },
        index=index,
    )

    encoded = arrow_payload.encode_arrow_dataframe(frame)
    restored = arrow_payload.decode_arrow_dataframe(encoded)

    assert encoded["format"] == arrow_payload.ARROW_DATAFRAME_FORMAT
    pd.testing.assert_frame_equal(restored, frame, check_categorical=True)


def test_dataframe_mapping_decodes_only_selected_frames():
    selected = pd.DataFrame({"value": [1, 2]})
    untouched = pd.DataFrame({"value": [3, 4]})
    packed = arrow_payload.pack_dataframe_mapping(
        {"selected": selected, "untouched": untouched, "status": "ok"},
        dataframe_keys=["selected"],
    )

    restored = arrow_payload.unpack_dataframe_mapping(
        packed,
        dataframe_keys=["selected"],
    )

    pd.testing.assert_frame_equal(restored["selected"], selected)
    assert restored["untouched"] is untouched
    assert restored["status"] == "ok"


def test_record_cube_arrow_round_trip_preserves_sparse_rows_and_scalars():
    records = {
        "Global": [
            {
                "value": np.float32(1.5),
                "nullable": pd.NA,
                "missing": None,
            },
            {"value": float("nan"), "date": pd.NaT},
        ],
        "Empty": [],
    }
    legacy_cube = snapshots.pack_record_mapping(records)

    restored_cube = arrow_payload.decode_arrow_record_cube(
        arrow_payload.encode_arrow_record_cube(legacy_cube)
    )
    restored = snapshots.unpack_record_mapping(restored_cube)

    assert list(restored) == ["Global", "Empty"]
    assert list(restored["Global"][0]) == list(records["Global"][0])
    assert isinstance(restored["Global"][0]["value"], np.float32)
    assert restored["Global"][0]["value"].tobytes() == (
        records["Global"][0]["value"].tobytes()
    )
    assert restored["Global"][0]["nullable"] is pd.NA
    assert restored["Global"][0]["missing"] is None
    assert np.isnan(restored["Global"][1]["value"])
    assert restored["Global"][1]["date"] is pd.NaT
    assert restored["Empty"] == []


def test_corrupt_arrow_payload_fails_closed():
    encoded = arrow_payload.encode_arrow_dataframe(
        pd.DataFrame({"value": [1, 2, 3]})
    )
    corrupt = copy.deepcopy(encoded)
    corrupt["payload"] = corrupt["payload"][:-8] + b"corrupt!"

    with pytest.raises(
        arrow_payload.ArrowPayloadError,
        match="corrupt|schema",
    ):
        arrow_payload.decode_arrow_dataframe(corrupt)
