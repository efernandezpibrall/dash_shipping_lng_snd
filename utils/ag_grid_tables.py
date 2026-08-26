"""Shared AG Grid builders for legacy Dash table migrations."""

from __future__ import annotations

import math
import re
from collections import OrderedDict
from typing import TYPE_CHECKING

import dash_ag_grid as dag

if TYPE_CHECKING:
    from typing import Any


AG_GRID_THEME = "ag-theme-alpine"
MCKINSEY_AG_GRID_CLASS = "mckinsey-ag-grid"

MCKINSEY_AG_GRID_DEFAULT_COL_DEF = {
    "sortable": True,
    "filter": False,
    "resizable": True,
    "wrapHeaderText": True,
    "autoHeaderHeight": True,
    "suppressHeaderMenuButton": True,
    "suppressHeaderFilterButton": True,
    "headerClass": "mckinsey-ag-grid-header",
    "cellClass": "mckinsey-ag-grid-cell",
}

MCKINSEY_AG_GRID_OPTIONS = {
    "animateRows": False,
    "enableCellTextSelection": True,
    "ensureDomOrder": True,
    "rowHeight": 32,
    "headerHeight": 36,
    "groupHeaderHeight": 30,
    "suppressRowHoverHighlight": False,
    "paginationPageSizeSelector": [10, 20, 50, 100],
}


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    try:
        return bool(value is not value)
    except Exception:
        return False


def _clean_value(value: Any) -> Any:
    if _is_blank(value):
        return None
    if isinstance(value, dict):
        return {key: _clean_value(nested_value) for key, nested_value in value.items()}
    if isinstance(value, list):
        return [_clean_value(item) for item in value]
    return value


def _clean_records(data: Any) -> list[dict[str, Any]]:
    if data is None:
        return []
    if hasattr(data, "to_dict"):
        data = data.to_dict("records")
    return [_clean_value(dict(row)) for row in list(data)]


def _format_specifier(column: dict[str, Any]) -> str | None:
    format_config = column.get("format")
    if format_config is None:
        return None

    if hasattr(format_config, "to_plotly_json"):
        try:
            format_config = format_config.to_plotly_json()
        except Exception:
            format_config = None

    if isinstance(format_config, dict):
        specifier = format_config.get("specifier")
        if specifier:
            return str(specifier)

    precision = column.get("precision")
    if precision is not None:
        return f",.{int(precision)}f"

    return None


def _numeric_value_formatter(specifier: str | None) -> dict[str, str]:
    specifier = specifier or ",.2f"
    return {
        "function": (
            "params.value !== null && params.value !== undefined && params.value !== '' "
            f"? d3.format('{specifier}')(Number(params.value)) : ''"
        )
    }


def _css_px(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    match = re.search(r"(-?\d+(?:\.\d+)?)", str(value))
    if not match:
        return None
    return int(float(match.group(1)))


def _dash_style_to_ag_style(style: dict[str, Any]) -> dict[str, Any]:
    excluded = {
        "height",
        "width",
        "minWidth",
        "maxWidth",
        "whiteSpace",
        "overflow",
        "overflowX",
        "overflowY",
    }
    return {key: value for key, value in style.items() if key not in excluded and key != "if"}


def _js_value(column_name: str) -> str:
    escaped = str(column_name).replace("\\", "\\\\").replace("'", "\\'")
    return f"(params.data ? params.data['{escaped}'] : undefined)"


def _filter_query_to_js(filter_query: str | None) -> str:
    if not filter_query:
        return "true"

    expression = str(filter_query)
    expression = re.sub(r"\{([^{}]+)\}", lambda match: _js_value(match.group(1)), expression)
    expression = expression.replace(" && ", " && ")
    expression = re.sub(r"\s+and\s+", " && ", expression, flags=re.IGNORECASE)
    expression = re.sub(r"\s+or\s+", " || ", expression, flags=re.IGNORECASE)
    expression = re.sub(r"(?<![<>=!])=(?!=)", "===", expression)
    return expression


def _row_index_condition(row_index: Any) -> str:
    if row_index == "odd":
        return "params.node && params.node.rowIndex % 2 === 1"
    if row_index == "even":
        return "params.node && params.node.rowIndex % 2 === 0"
    if isinstance(row_index, int):
        return f"params.node && params.node.rowIndex === {row_index}"
    return "true"


def _condition_to_js(condition: dict[str, Any] | None) -> str:
    if not condition:
        return "true"

    clauses: list[str] = []
    if "filter_query_js" in condition:
        clauses.append(str(condition.get("filter_query_js")))
    if "filter_query" in condition:
        clauses.append(_filter_query_to_js(condition.get("filter_query")))
    if "row_index" in condition:
        clauses.append(_row_index_condition(condition.get("row_index")))
    return " && ".join(f"({clause})" for clause in clauses) if clauses else "true"


def _merge_cell_style(column_def: dict[str, Any], style: dict[str, Any], condition: str | None = None) -> None:
    ag_style = _dash_style_to_ag_style(style)
    if not ag_style:
        return

    existing = column_def.get("cellStyle")
    if condition and condition != "true":
        if not isinstance(existing, dict) or "styleConditions" not in existing:
            existing = {"styleConditions": [], "defaultStyle": existing if isinstance(existing, dict) else {}}
        existing["styleConditions"].append({"condition": condition, "style": ag_style})
        column_def["cellStyle"] = existing
        return

    if isinstance(existing, dict) and "styleConditions" in existing:
        existing["defaultStyle"] = {**existing.get("defaultStyle", {}), **ag_style}
        column_def["cellStyle"] = existing
    elif isinstance(existing, dict):
        column_def["cellStyle"] = {**existing, **ag_style}
    else:
        column_def["cellStyle"] = ag_style


def _apply_column_style_rules(
    column_defs_by_field: dict[str, dict[str, Any]],
    style_cell_conditional: list[dict[str, Any]] | None,
) -> None:
    for rule in style_cell_conditional or []:
        condition = rule.get("if", {})
        column_ids = condition.get("column_id")
        if column_ids is None:
            continue
        if not isinstance(column_ids, (list, tuple, set)):
            column_ids = [column_ids]

        for column_id in column_ids:
            column_def = column_defs_by_field.get(str(column_id))
            if not column_def:
                continue

            width = _css_px(rule.get("width"))
            min_width = _css_px(rule.get("minWidth"))
            max_width = _css_px(rule.get("maxWidth"))
            if width:
                column_def["width"] = width
            if min_width:
                column_def["minWidth"] = min_width
            if max_width:
                column_def["maxWidth"] = max_width

            _merge_cell_style(column_def, rule)


def _apply_conditional_style_rules(
    column_defs_by_field: dict[str, dict[str, Any]],
    style_data_conditional: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    row_style_conditions: list[dict[str, Any]] = []

    for rule in style_data_conditional or []:
        condition = rule.get("if", {})
        style = _dash_style_to_ag_style(rule)
        if not style:
            continue

        column_ids = condition.get("column_id")
        condition_js = _condition_to_js(condition)

        if column_ids is not None:
            if not isinstance(column_ids, (list, tuple, set)):
                column_ids = [column_ids]
            for column_id in column_ids:
                column_def = column_defs_by_field.get(str(column_id))
                if column_def:
                    _merge_cell_style(column_def, style, condition_js)
        else:
            row_style_conditions.append({"condition": condition_js, "style": style})

    return row_style_conditions


def _column_header_path(column: dict[str, Any]) -> list[str]:
    name = column.get("name", column.get("id", ""))
    if isinstance(name, (list, tuple)):
        return [str(part) for part in name]
    return [str(name)]


def _build_grouped_column_defs(columns: list[dict[str, Any]], leaf_defs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not any(len(_column_header_path(column)) > 1 for column in columns):
        return leaf_defs

    root: OrderedDict[str, Any] = OrderedDict()
    for column, leaf_def in zip(columns, leaf_defs):
        path = _column_header_path(column)
        if len(path) <= 1:
            root[leaf_def["field"]] = leaf_def
            continue

        current = root
        for group_name in path[:-1]:
            group = current.setdefault(group_name, OrderedDict())
            current = group
        leaf_def["headerName"] = path[-1] or leaf_def.get("field", "")
        current[leaf_def["field"]] = leaf_def

    def materialize(node: OrderedDict[str, Any]) -> list[dict[str, Any]]:
        output = []
        for key, value in node.items():
            if isinstance(value, OrderedDict):
                output.append(
                    {
                        "headerName": key,
                        "children": materialize(value),
                        "marryChildren": True,
                    }
                )
            else:
                output.append(value)
        return output

    return materialize(root)


def _build_column_defs(
    columns: list[dict[str, Any]],
    *,
    editable: bool = False,
    sort_action: str | None = "native",
    filter_action: str | None = "none",
    fixed_columns: dict[str, Any] | None = None,
    hidden_columns: list[str] | None = None,
    style_cell_conditional: list[dict[str, Any]] | None = None,
    style_data_conditional: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    hidden_set = {str(column) for column in (hidden_columns or [])}
    pinned_count = 0
    if isinstance(fixed_columns, dict):
        pinned_count = int(fixed_columns.get("data") or 0)

    leaf_defs: list[dict[str, Any]] = []
    column_defs_by_field: dict[str, dict[str, Any]] = {}

    for index, column in enumerate(columns or []):
        field = str(column.get("id", column.get("name", "")))
        if not field:
            continue

        is_numeric = column.get("type") == "numeric"
        is_editable = bool(column.get("editable", editable))
        header_name = _column_header_path(column)[-1]
        base_cell_class = (
            "mckinsey-ag-grid-cell mckinsey-ag-grid-number-cell"
            if is_numeric
            else "mckinsey-ag-grid-cell mckinsey-ag-grid-text-cell"
        )
        extra_cell_class = str(column.get("cellClass") or "").strip()
        extra_header_class = str(column.get("headerClass") or "").strip()
        column_def: dict[str, Any] = {
            "headerName": header_name,
            "field": field,
            "sortable": bool(column.get("sortable", sort_action != "none")),
            "filter": (
                "agNumberColumnFilter"
                if filter_action == "native" and is_numeric
                else "agTextColumnFilter"
                if filter_action == "native"
                else False
            ),
            "resizable": True,
            "editable": is_editable,
            "hide": field in hidden_set,
            "floatingFilter": filter_action == "native",
            "cellClass": f"{base_cell_class} {extra_cell_class}".strip(),
            "headerClass": f"mckinsey-ag-grid-header {extra_header_class}".strip(),
        }

        if is_numeric:
            column_def["type"] = "rightAligned"
            column_def["valueFormatter"] = _numeric_value_formatter(_format_specifier(column))

        if index < pinned_count and field not in hidden_set:
            column_def.update({"pinned": "left", "lockPinned": True, "suppressMovable": True})

        if is_editable:
            column_def["cellClass"] = f"{column_def['cellClass']} mckinsey-ag-grid-editable-cell"

        for prop_name in ("cellRenderer", "cellStyle", "tooltipValueGetter", "valueGetter"):
            if prop_name in column:
                column_def[prop_name] = column[prop_name]

        cell_class_rules = column.get("cellClassRules")
        if isinstance(cell_class_rules, dict) and cell_class_rules:
            column_def["cellClassRules"] = {
                class_name: (
                    rule.get("function")
                    if isinstance(rule, dict) and "function" in rule
                    else rule
                )
                for class_name, rule in cell_class_rules.items()
            }

        column_defs_by_field[field] = column_def
        leaf_defs.append(column_def)

    _apply_column_style_rules(column_defs_by_field, style_cell_conditional)
    row_style_conditions = _apply_conditional_style_rules(column_defs_by_field, style_data_conditional)
    return _build_grouped_column_defs(columns or [], leaf_defs), row_style_conditions


def _default_height(row_count: int, page_size: int | None, page_action: str | None, style_table: dict[str, Any] | None) -> str:
    style_table = style_table or {}
    explicit_height = style_table.get("height") or style_table.get("maxHeight")
    if explicit_height:
        return str(explicit_height)

    visible_rows = row_count
    if page_action != "none" and page_size:
        visible_rows = min(row_count or page_size, page_size)
    else:
        visible_rows = min(max(row_count, 6), 18)

    return f"{max(180, min(720, 48 + visible_rows * 32))}px"


def create_ag_grid_from_datatable(
    *,
    id: Any = None,
    columns: list[dict[str, Any]] | None = None,
    data: Any = None,
    editable: bool = False,
    sort_action: str | None = "native",
    filter_action: str | None = "none",
    page_action: str | None = "native",
    page_size: int | None = None,
    sort_by: list[dict[str, Any]] | None = None,
    fixed_columns: dict[str, Any] | None = None,
    style_table: dict[str, Any] | None = None,
    style_cell_conditional: list[dict[str, Any]] | None = None,
    style_data_conditional: list[dict[str, Any]] | None = None,
    hidden_columns: list[str] | None = None,
    fill_width: bool = True,
    export_format: str | None = None,
    row_selectable: str | None = None,
    className: str = "",
    height: str | int | None = None,
    columnSize: str | None = None,
    dashGridOptions: dict[str, Any] | None = None,
    defaultColDef: dict[str, Any] | None = None,
    getRowStyle: dict[str, Any] | None = None,
    rowClassRules: dict[str, str] | None = None,
    getRowId: str | dict[str, str] | None = None,
    **_: Any,
) -> dag.AgGrid:
    row_data = _clean_records(data)
    column_defs, row_style_conditions = _build_column_defs(
        columns or [],
        editable=editable,
        sort_action=sort_action,
        filter_action=filter_action,
        fixed_columns=fixed_columns,
        hidden_columns=hidden_columns,
        style_cell_conditional=style_cell_conditional,
        style_data_conditional=style_data_conditional,
    )

    grid_options = {
        **MCKINSEY_AG_GRID_OPTIONS,
        **(dashGridOptions or {}),
    }
    resolved_default_col_def = {**MCKINSEY_AG_GRID_DEFAULT_COL_DEF, **(defaultColDef or {})}
    if filter_action == "native":
        resolved_default_col_def.update(
            {
                "floatingFilter": True,
                "suppressHeaderMenuButton": False,
                "suppressHeaderFilterButton": False,
            }
        )
        grid_options["floatingFiltersHeight"] = 30
    if page_action != "none" and page_size:
        grid_options.update({"pagination": True, "paginationPageSize": page_size})
    else:
        grid_options["pagination"] = False
        grid_options.pop("paginationPageSizeSelector", None)

    if editable:
        grid_options.update(
            {
                "singleClickEdit": True,
                "undoRedoCellEditing": True,
                "undoRedoCellEditingLimit": 30,
                "stopEditingWhenCellsLoseFocus": True,
            }
        )

    if row_selectable:
        is_multi_select = row_selectable == "multi"
        grid_options["rowSelection"] = {
            "mode": "multiRow" if is_multi_select else "singleRow",
            "checkboxes": False,
            "enableClickSelection": True,
        }
        if is_multi_select:
            grid_options["rowSelection"].update(
                {
                    "headerCheckbox": False,
                    "enableSelectionWithoutKeys": True,
                }
            )

    if sort_by:
        sort_index_by_field = {rule.get("column_id"): i for i, rule in enumerate(sort_by)}
        for definition in iter_leaf_column_defs(column_defs):
            field = definition.get("field")
            if field not in sort_index_by_field:
                continue
            rule = sort_by[sort_index_by_field[field]]
            definition["sort"] = "desc" if rule.get("direction") == "desc" else "asc"
            definition["sortIndex"] = sort_index_by_field[field]

    grid_kwargs: dict[str, Any] = {
        "id": id,
        "rowData": row_data,
        "columnDefs": column_defs,
        "defaultColDef": resolved_default_col_def,
        "dashGridOptions": grid_options,
        "className": f"{AG_GRID_THEME} {MCKINSEY_AG_GRID_CLASS} {className}".strip(),
        "style": {"width": "100%", "height": str(height or _default_height(len(row_data), page_size, page_action, style_table))},
        "dangerously_allow_code": True,
    }
    if getRowId:
        grid_kwargs["getRowId"] = getRowId

    if fill_width:
        grid_kwargs["columnSize"] = columnSize or "responsiveSizeToFit"
    elif columnSize:
        grid_kwargs["columnSize"] = columnSize

    if export_format:
        grid_kwargs["csvExportParams"] = {"fileName": f"{id or 'table'}.csv"}
        grid_kwargs["exportDataAsCsv"] = False

    if row_style_conditions:
        grid_kwargs["getRowStyle"] = {
            "styleConditions": row_style_conditions,
            "defaultStyle": {},
        }
    if getRowStyle:
        if "getRowStyle" in grid_kwargs and isinstance(getRowStyle, dict):
            grid_kwargs["getRowStyle"]["styleConditions"].extend(getRowStyle.get("styleConditions", []))
            if getRowStyle.get("defaultStyle"):
                grid_kwargs["getRowStyle"]["defaultStyle"] = getRowStyle["defaultStyle"]
        else:
            grid_kwargs["getRowStyle"] = getRowStyle
    if rowClassRules:
        grid_kwargs["rowClassRules"] = rowClassRules

    return dag.AgGrid(**grid_kwargs)


def iter_leaf_column_defs(column_defs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    leaves: list[dict[str, Any]] = []
    for column_def in column_defs or []:
        children = column_def.get("children")
        if children:
            leaves.extend(iter_leaf_column_defs(children))
        else:
            leaves.append(column_def)
    return leaves


def datatable_columns_to_ag_grid_column_defs(
    columns: list[dict[str, Any]] | None,
    *,
    editable: bool = False,
    sort_action: str | None = "native",
    filter_action: str | None = "none",
    hidden_columns: list[str] | None = None,
) -> list[dict[str, Any]]:
    column_defs, _ = _build_column_defs(
        columns or [],
        editable=editable,
        sort_action=sort_action,
        filter_action=filter_action,
        hidden_columns=hidden_columns,
    )
    return column_defs


def ag_grid_column_defs_to_datatable_columns(column_defs: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    columns = []
    for column_def in iter_leaf_column_defs(column_defs or []):
        if column_def.get("hide"):
            continue
        field = column_def.get("field")
        if not field:
            continue
        columns.append({"name": column_def.get("headerName") or field, "id": field})
    return columns


def ag_grid_cell_clicked_to_active_cell(cell_clicked: dict[str, Any] | None) -> dict[str, Any] | None:
    if not cell_clicked:
        return None
    column_id = cell_clicked.get("colId") or cell_clicked.get("column", {}).get("colId")
    row_index = cell_clicked.get("rowIndex")
    if row_index is None:
        row_index = cell_clicked.get("row")
    if column_id is None or row_index is None:
        return None
    return {"row": row_index, "column_id": column_id, "data": cell_clicked.get("data")}
