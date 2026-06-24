"""Shared matrix helpers for LNG balance demand/supply pages."""

from io import BytesIO

import pandas as pd


def align_matrix_to_reference_months(
    matrix_df: pd.DataFrame,
    reference_month_labels: list[str],
) -> pd.DataFrame:
    reference_index = (
        pd.Series(reference_month_labels, dtype="object")
        .pipe(pd.to_datetime, errors="coerce")
        .dt.to_period("M")
        .dt.to_timestamp()
    )
    reference_index = pd.Index(reference_index.dropna().unique())

    numeric_columns = [column for column in matrix_df.columns if column != "Month"]
    if reference_index.empty:
        return pd.DataFrame(columns=["Month"] + numeric_columns)

    if matrix_df.empty:
        aligned_df = pd.DataFrame(index=reference_index)
    else:
        aligned_df = matrix_df.copy()
        aligned_df["Month"] = pd.to_datetime(
            aligned_df["Month"].astype(str),
            errors="coerce",
        ).dt.to_period("M").dt.to_timestamp()
        aligned_df = (
            aligned_df.dropna(subset=["Month"])
            .drop_duplicates(subset=["Month"], keep="last")
            .set_index("Month")
            .sort_index()
            .reindex(reference_index)
        )

    for column_name in numeric_columns:
        source_series = (
            aligned_df[column_name]
            if column_name in aligned_df.columns
            else pd.Series(float("nan"), index=aligned_df.index, dtype="float64")
        )
        aligned_df[column_name] = pd.to_numeric(
            source_series,
            errors="coerce",
        )

    aligned_df.index.name = "Month"
    result_df = aligned_df.reset_index()
    result_df["Month"] = pd.to_datetime(result_df["Month"]).dt.strftime("%Y-%m")

    return result_df[["Month"] + numeric_columns]


def build_delta_matrix(
    baseline_matrix: pd.DataFrame,
    comparison_matrix: pd.DataFrame,
) -> pd.DataFrame:
    if baseline_matrix.empty:
        return pd.DataFrame(columns=["Month", "Total MMTPA"])

    numeric_columns = [column for column in baseline_matrix.columns if column != "Month"]
    delta_df = baseline_matrix.copy()
    delta_df[numeric_columns] = delta_df[numeric_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    comparison_aligned = comparison_matrix.copy()
    for column in numeric_columns:
        if column not in comparison_aligned.columns:
            comparison_aligned[column] = float("nan")

    comparison_aligned = comparison_aligned[["Month"] + numeric_columns]
    comparison_aligned[numeric_columns] = comparison_aligned[numeric_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    comparison_aligned = comparison_aligned.set_index("Month").reindex(delta_df["Month"])

    delta_index = delta_df.set_index("Month")
    delta_index[numeric_columns] = (
        delta_index[numeric_columns] - comparison_aligned[numeric_columns]
    ).round(2)

    return delta_index.reset_index()


def export_matrix_to_excel_bytes(df: pd.DataFrame, sheet_name: str) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)

        worksheet = writer.sheets[sheet_name]
        for column_cells in worksheet.columns:
            max_length = 0
            column_letter = column_cells[0].column_letter
            for cell in column_cells:
                cell_value = "" if cell.value is None else str(cell.value)
                if len(cell_value) > max_length:
                    max_length = len(cell_value)
            worksheet.column_dimensions[column_letter].width = min(max_length + 2, 24)

    output.seek(0)
    return output.getvalue()
