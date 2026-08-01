"""Versioned Arrow IPC payloads for large immutable dashboard frames."""

from __future__ import annotations

from contextlib import suppress
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as pa_ipc


ARROW_DATAFRAME_FORMAT = "dashboard-arrow-dataframe-ipc-zstd-v1"
ARROW_RECORD_CUBE_FORMAT = "entity-record-cube-arrow-ipc-zstd-v1"
_RECORD_CUBE_FORMAT = "entity_record_cube_v1"
_PRESENT_COLUMNS_FIELD = "__dashboard_present_columns__"


class ArrowPayloadError(ValueError):
    """Raised when an Arrow payload is malformed or cannot be restored."""


def _record_cube_special_tag(value: Any) -> Any:
    if value is None:
        return "none"
    if value is pd.NA:
        return "pd_na"
    if value is pd.NaT:
        return "nat"
    if isinstance(value, np.generic):
        return {
            "type": "numpy_scalar",
            "dtype": value.dtype.str,
            "bytes": value.tobytes().hex(),
        }
    if isinstance(value, float):
        if math.isnan(float(value)):
            return "nan"
        if math.isinf(float(value)):
            return "positive_infinity" if value > 0 else "negative_infinity"
    return None


def _restore_record_cube_special(tag: Any) -> Any:
    if isinstance(tag, Mapping) and tag.get("type") == "numpy_scalar":
        try:
            return np.frombuffer(
                bytes.fromhex(str(tag["bytes"])),
                dtype=np.dtype(str(tag["dtype"])),
                count=1,
            )[0]
        except Exception as exc:
            raise ArrowPayloadError(
                "Arrow record cube numpy scalar tag is invalid"
            ) from exc
    values = {
        "none": None,
        "pd_na": pd.NA,
        "nat": pd.NaT,
        "nan": float("nan"),
        "positive_infinity": float("inf"),
        "negative_infinity": float("-inf"),
    }
    try:
        return values[tag]
    except KeyError as exc:
        raise ArrowPayloadError(
            f"Arrow record cube special-value tag is invalid: {tag!r}"
        ) from exc


def _dictionary_encode_strings(table: pa.Table) -> pa.Table:
    arrays = []
    names = []
    for name, column in zip(table.column_names, table.columns, strict=True):
        if pa.types.is_string(column.type) or pa.types.is_large_string(
            column.type
        ):
            column = column.dictionary_encode()
        arrays.append(column)
        names.append(name)
    return pa.Table.from_arrays(
        arrays,
        names=names,
        metadata=table.schema.metadata,
    )


def encode_arrow_dataframe(frame: pd.DataFrame) -> dict[str, Any]:
    """Encode one DataFrame as Arrow IPC while preserving pandas metadata."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Arrow dataframe encoding requires a pandas DataFrame")
    try:
        table = pa.Table.from_pandas(frame, preserve_index=True, safe=True)
        table = _dictionary_encode_strings(table)
        sink = pa.BufferOutputStream()
        options = pa_ipc.IpcWriteOptions(compression="zstd")
        with pa_ipc.new_stream(sink, table.schema, options=options) as writer:
            writer.write_table(table)
        payload = sink.getvalue().to_pybytes()
    except Exception as exc:
        raise ArrowPayloadError("DataFrame could not be encoded as Arrow") from exc
    return {
        "format": ARROW_DATAFRAME_FORMAT,
        "payload": payload,
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "column_labels": list(frame.columns),
        "dtypes": [str(dtype) for dtype in frame.dtypes],
        "schema_sha256": hashlib.sha256(
            table.schema.serialize().to_pybytes()
        ).hexdigest(),
    }


def decode_arrow_dataframe(value: Any) -> pd.DataFrame:
    """Restore one exact pandas DataFrame from a versioned Arrow envelope."""

    if not (
        isinstance(value, Mapping)
        and value.get("format") == ARROW_DATAFRAME_FORMAT
    ):
        if isinstance(value, pd.DataFrame):
            return value
        raise ArrowPayloadError("Arrow dataframe payload format is invalid")
    payload = value.get("payload")
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise ArrowPayloadError("Arrow dataframe payload bytes are missing")
    try:
        with pa_ipc.open_stream(pa.py_buffer(bytes(payload))) as reader:
            table = reader.read_all()
        schema_sha256 = hashlib.sha256(
            table.schema.serialize().to_pybytes()
        ).hexdigest()
        if schema_sha256 != value.get("schema_sha256"):
            raise ArrowPayloadError("Arrow dataframe schema does not match")
        frame = table.to_pandas()
        labels = list(value.get("column_labels") or [])
        if labels and len(labels) == len(frame.columns):
            frame.columns = labels
        for column, dtype_name in zip(
            frame.columns,
            value.get("dtypes") or [],
            strict=False,
        ):
            if str(frame[column].dtype) == str(dtype_name):
                continue
            with suppress(TypeError, ValueError):
                frame[column] = frame[column].astype(dtype_name)
        if len(frame) != int(value.get("rows")):
            raise ArrowPayloadError("Arrow dataframe row count does not match")
        if len(frame.columns) != int(value.get("columns")):
            raise ArrowPayloadError("Arrow dataframe column count does not match")
        return frame
    except ArrowPayloadError:
        raise
    except Exception as exc:
        raise ArrowPayloadError("Arrow dataframe payload is corrupt") from exc


def pack_dataframe_mapping(
    value: Mapping[str, Any],
    *,
    dataframe_keys: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Arrow-pack selected top-level DataFrames in one immutable mapping."""

    allowed = set(dataframe_keys) if dataframe_keys is not None else None
    result: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, pd.DataFrame) and (
            allowed is None or key in allowed
        ):
            result[key] = encode_arrow_dataframe(item)
        else:
            result[key] = item
    return result


def unpack_dataframe_mapping(
    value: Mapping[str, Any],
    *,
    dataframe_keys: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Restore selected Arrow DataFrames while leaving other values intact."""

    allowed = set(dataframe_keys) if dataframe_keys is not None else None
    result: dict[str, Any] = {}
    for key, item in value.items():
        should_decode = allowed is None or key in allowed
        if (
            should_decode
            and isinstance(item, Mapping)
            and item.get("format") == ARROW_DATAFRAME_FORMAT
        ):
            result[key] = decode_arrow_dataframe(item)
        else:
            result[key] = item
    return result


def encode_arrow_record_cube(value: Any) -> dict[str, Any]:
    """Encode an ordered entity-record cube as one Arrow table."""

    if not (
        isinstance(value, Mapping)
        and value.get("format") == _RECORD_CUBE_FORMAT
    ):
        raise ArrowPayloadError("Record cube format is invalid")
    columns = list(value.get("columns") or [])
    rows = list(value.get("rows") or [])
    present_columns = list(value.get("present_columns") or [])
    if len(present_columns) != len(rows):
        present_columns = [list(range(len(columns))) for _ in rows]
    special_values = []
    for row_index, (row, presence) in enumerate(
        zip(rows, present_columns, strict=True)
    ):
        for column_index in presence or []:
            tag = _record_cube_special_tag(row[column_index])
            if tag is not None:
                special_values.append(
                    [row_index, int(column_index), tag]
                )
    try:
        frame = pd.DataFrame(rows, columns=columns)
        frame[_PRESENT_COLUMNS_FIELD] = [
            list(indices or []) for indices in present_columns
        ]
        encoded = encode_arrow_dataframe(frame)
    except Exception as exc:
        raise ArrowPayloadError("Record cube could not be Arrow encoded") from exc
    return {
        "format": ARROW_RECORD_CUBE_FORMAT,
        "entities": list(value.get("entities") or []),
        "columns": columns,
        "entity_row_counts": list(value.get("entity_row_counts") or []),
        "special_values": special_values,
        "frame": encoded,
    }


def decode_arrow_record_cube(value: Any) -> Any:
    """Restore the legacy ordered record-cube contract from Arrow."""

    if not (
        isinstance(value, Mapping)
        and value.get("format") == ARROW_RECORD_CUBE_FORMAT
    ):
        return value
    columns = list(value.get("columns") or [])
    frame = decode_arrow_dataframe(value.get("frame"))
    if _PRESENT_COLUMNS_FIELD not in frame:
        raise ArrowPayloadError("Arrow record cube has no presence column")
    present_columns = [
        [
            int(index)
            for index in (
                [] if indices is None else list(indices)
            )
        ]
        for indices in frame.pop(_PRESENT_COLUMNS_FIELD).tolist()
    ]
    missing_columns = [column for column in columns if column not in frame]
    if missing_columns:
        raise ArrowPayloadError(
            f"Arrow record cube is missing columns: {missing_columns}"
        )
    rows = frame.loc[:, columns].values.tolist()
    for raw_row_index, raw_column_index, tag in (
        value.get("special_values") or []
    ):
        row_index = int(raw_row_index)
        column_index = int(raw_column_index)
        if not (
            0 <= row_index < len(rows)
            and 0 <= column_index < len(columns)
        ):
            raise ArrowPayloadError(
                "Arrow record cube special-value position is invalid"
            )
        rows[row_index][column_index] = _restore_record_cube_special(tag)
    return {
        "format": _RECORD_CUBE_FORMAT,
        "entities": list(value.get("entities") or []),
        "columns": columns,
        "entity_row_counts": list(value.get("entity_row_counts") or []),
        "rows": rows,
        "present_columns": present_columns,
    }
