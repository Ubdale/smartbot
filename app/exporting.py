"""Styled Excel workbooks of an answer's table (the table only: charts export as PNG)."""

import io
import re
from datetime import date, datetime
from typing import Any

from openpyxl import Workbook
from openpyxl.formatting.rule import DataBarRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet

from app.styling import ExportStyle, apply_style, color

BRAND = "4F46E5"  # indigo, as in the app
BAND = "EEF2FF"
GRID = Side(style="thin", color="E4E4E7")
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_ISO_DATETIME = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}")
_FORMULA = re.compile(r"^[=+\-@]")
# Identifiers are numbers you never add up: no thousands separators for them.
_ID_LIKE = re.compile(r"(^|[\s_\-])(id|key|code|no|number|num|zip|phone|year|ref)($|[\s_\-])", re.I)


def _cell_value(v: Any) -> Any:
    """Real dates/numbers for Excel; text that looks like a formula is kept as text."""
    if isinstance(v, str):
        if _ISO_DATE.match(v):
            return date.fromisoformat(v)
        if _ISO_DATETIME.match(v):
            try:
                return datetime.fromisoformat(v)
            except ValueError:
                return v
        if _FORMULA.match(v):
            return "'" + v  # never let a cell from the data run as a formula
    return v


def _pretty(name: str) -> str:
    m = re.match(r"^(count_distinct|count|sum|avg|min|max)_(.+)$", name)
    if name == "count":
        return "Count"
    if not m:
        return name
    label = {"count_distinct": "Distinct", "count": "Count of", "sum": "Total", "avg": "Average",
             "min": "Minimum", "max": "Maximum"}[m[1]]  # fmt: skip
    return f"{label} {m[2].replace('_', ' ')}"


def is_id_like(column: str) -> bool:
    return bool(_ID_LIKE.search(column))


def _number_format(values: list[Any], column: str = "") -> str | None:
    nums = [v for v in values if isinstance(v, int | float) and not isinstance(v, bool)]
    if not nums or len(nums) < len([v for v in values if v is not None]):
        return None
    if is_id_like(column) and all(float(v).is_integer() for v in nums):
        return "0"
    if all(float(v).is_integer() for v in nums):
        return "#,##0"
    return "#,##0.00"


_METRIC_TITLE = {"sum": "Total", "avg": "Average", "min": "Lowest", "max": "Highest"}
_OP_TITLE = {"=": "is", "!=": "is not", ">": "over", ">=": "at least", "<": "under",
             "<=": "at most", "in": "is", "contains": "contains", "between": "between"}  # fmt: skip


def _name(column: str) -> str:
    return re.sub(r"[_\s]+", " ", column).strip()


def _value(v: Any) -> str:
    if isinstance(v, list):
        shown = [_value(x) for x in v[:3]]
        return " or ".join(shown) + (" ..." if len(v) > 3 else "")
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    if isinstance(v, int) and not isinstance(v, bool):
        return f"{v:,}"
    return str(v)


def export_title(explanation: dict[str, Any] | None, dataset_name: str = "") -> str:
    """A title for downloads, written by code from what was computed (never the user's
    message, which may be chatty, Roman Urdu or a question): "Total revenue by region
    (2025)", "Rows where Assignee is Sara Ahmed", "Columns of Task go for QC"."""
    explanation = explanation or {}
    plan = explanation.get("plan") or {}
    metric = plan.get("metric")
    if metric == "describe":
        return f"Columns of {dataset_name}" if dataset_name else "Columns of the data"
    if not metric:
        # Reports computed without a plan (data doctor, what changed): their description.
        d = re.sub(r";\s*(showing up to|sorted by).*$", "", explanation.get("description", ""))
        return d.rstrip(". ")[:120] or dataset_name or "AskData answer"

    col = _name(plan.get("metric_column") or "")
    if metric == "rows":
        what = "Rows"
    elif metric == "count":
        what = "Count"
    elif metric == "count_distinct":
        what = f"Distinct {col}"
    else:
        what = f"{_METRIC_TITLE.get(metric, metric)} {col}".strip()

    groups = [_name(g) for g in plan.get("group_by") or []]
    if plan.get("time_grain"):
        groups.insert(0, str(plan["time_grain"]))
    sort, limit = plan.get("sort") or {}, int(plan.get("limit") or 0)
    # "Top 5 customers" only when asked for: 20 is just the default number of rows.
    if groups and sort.get("by") == "value" and limit and limit < 20:
        top = "Top" if sort.get("direction", "desc") == "desc" else "Bottom"
        title = f"{top} {limit} {' and '.join(groups)} by {what[0].lower() + what[1:]}"
    else:
        title = what + (f" by {' and '.join(groups)}" if groups else "")

    conditions = []
    for f in (plan.get("filters") or [])[:2]:
        column, op, value = _name(f.get("column", "")), f.get("op"), f.get("value")
        if op == "about":
            conditions.append(f"{column} about \u201c{value}\u201d")
        elif value is None:
            conditions.append(f"{column} {'is empty' if op == '=' else 'is filled in'}")
        elif op == "between" and isinstance(value, list) and len(value) == 2:
            conditions.append(f"{column} between {_value(value[0])} and {_value(value[1])}")
        else:
            conditions.append(f"{column} {_OP_TITLE.get(op, op)} {_value(value)}")
    if conditions:
        more = " ..." if len(plan.get("filters") or []) > 2 else ""
        title += " where " + " and ".join(conditions) + more
    period = (explanation.get("date_range") or {}).get("label")
    if period:
        title += f" ({period})"
    return title[:1].upper() + title[1:150]


def _sheet_name(text: str) -> str:
    return re.sub(r"[\[\]\*\?/\\:]", " ", text)[:31].strip() or "Answer"


def build_answer_workbook(
    *,
    question: str,
    summary: str,
    table: dict[str, Any],
    chart: dict[str, Any] | None,
    dataset_name: str,
    created_at: datetime,
    explanation: dict[str, Any] | None = None,
    style: ExportStyle | None = None,
    today: date | None = None,
) -> bytes:
    columns: list[str] = table.get("columns", [])
    rows: list[list[Any]] = table.get("rows", [])
    explanation = explanation or {}
    title = export_title(explanation, dataset_name)
    plan = explanation.get("plan") or {}

    # Formatting the user asked for: applied by code (sort, hide, highlight, summaries).
    style = style or ExportStyle()
    summable = summable_columns(columns, rows)
    styled = apply_style(columns, rows, style, today or date.today(), numeric_columns=summable)
    columns, rows = styled.columns, styled.rows
    title = style.title or title
    header_fill, header_text = styled.header or (BRAND, "FFFFFF")

    wb = Workbook()
    ws: Worksheet = wb.active  # type: ignore[assignment]
    ws.title = _sheet_name(title)
    ws.sheet_view.showGridLines = False

    last_col = get_column_letter(max(len(columns), 1))

    # Just the table: headers in row 1, data from row 2.
    header_row = 1
    for j, name in enumerate(columns, start=1):
        cell = ws.cell(row=header_row, column=j, value=_pretty(name))
        cell.font = Font(bold=True, color=header_text)
        cell.fill = PatternFill("solid", fgColor=header_fill)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = Border(bottom=GRID)
    ws.row_dimensions[header_row].height = 22

    formats = [_number_format([r[j] for r in rows], columns[j]) for j in range(len(columns))]
    for i, row in enumerate(rows, start=header_row + 1):
        mark = color(styled.row_colors[i - header_row - 1], strong=False)
        fill: PatternFill | None = None
        if mark:
            fill = PatternFill("solid", fgColor=mark[0])
        elif style.banded_rows and (i - header_row) % 2 == 0:
            fill = PatternFill("solid", fgColor=BAND)
        for j, value in enumerate(row, start=1):
            cell = ws.cell(row=i, column=j, value=_cell_value(value))
            cell.border = Border(bottom=GRID)
            if fill:
                cell.fill = fill
            if mark:
                cell.font = Font(color=mark[1])
            fmt = formats[j - 1]
            if fmt and isinstance(cell.value, int | float):
                cell.number_format = fmt
                cell.alignment = Alignment(horizontal="right")
            elif isinstance(cell.value, datetime):
                cell.number_format = "yyyy-mm-dd hh:mm"
            elif isinstance(cell.value, date):
                cell.number_format = "yyyy-mm-dd"

    last_row = header_row + max(len(rows), 1)
    if rows:
        ws.auto_filter.ref = f"A{header_row}:{last_col}{last_row}"

    # Grouped totals/counts: a live Totals row (Excel formulas) and data bars.
    metric = plan.get("metric")
    value_col = len(columns)
    if style.summary_rows is not None:
        _summary_rows(ws, style, columns, summable, formats, header_row, last_row)
    if (
        rows
        and metric in ("sum", "count", "count_distinct")
        and len(columns) > 1
        and (formats[-1] not in (None, "0"))
    ):
        letter = get_column_letter(value_col)
        ws.conditional_formatting.add(
            f"{letter}{header_row + 1}:{letter}{last_row}",
            DataBarRule(start_type="min", end_type="max", color="A5B4FC"),  # type: ignore[no-untyped-call]
        )
    ws.freeze_panes = f"A{header_row + 1}"

    # Column widths from the content (capped so long text wraps instead).
    for j, name in enumerate(columns, start=1):
        # +5 leaves room for the filter arrow next to the header text.
        longest = max([len(_pretty(name)) + 5] + [len(str(r[j - 1])) + 2 for r in rows[:500]])
        ws.column_dimensions[get_column_letter(j)].width = min(max(10, longest), 60)

    ws.print_title_rows = f"{header_row}:{header_row}"
    ws.page_setup.orientation = "landscape" if len(columns) > 4 else "portrait"
    ws.page_setup.fitToWidth = 1

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


_SUMMARY_FORMULA = {"total": "SUM", "average": "AVERAGE", "min": "MIN", "max": "MAX",
                    "count": "COUNT"}  # fmt: skip


def summable_columns(columns: list[str], rows: list[list[Any]]) -> list[str]:
    """Numeric columns it makes sense to total or average (not IDs or years)."""
    return [
        c
        for j, c in enumerate(columns)
        if _number_format([r[j] for r in rows], c) not in (None, "0")
    ]


def _summary_rows(
    ws: Worksheet,
    style: ExportStyle,
    columns: list[str],
    summable: list[str],
    formats: list[str | None],
    header_row: int,
    last_row: int,
) -> None:
    """Rows of live Excel formulas under the table: =AVERAGE(...), =SUM(...), ..."""
    for k, kind in enumerate(style.summary_rows or []):
        r = last_row + 1 + k
        label_col = next((j for j, c in enumerate(columns, 1) if c not in summable), None)
        for j, c in enumerate(columns, start=1):
            cell = ws.cell(row=r, column=j)
            if k == 0:
                cell.border = Border(top=Side(style="medium", color=BRAND))
            if c in summable:
                letter = get_column_letter(j)
                func = _SUMMARY_FORMULA[kind]
                cell.value = f"={func}({letter}{header_row + 1}:{letter}{last_row})"
                cell.number_format = (
                    "#,##0" if kind == "count" else
                    "#,##0.00" if kind == "average" else formats[j - 1] or "#,##0"
                )  # fmt: skip
                cell.alignment = Alignment(horizontal="right")
                cell.font = Font(bold=True)
        if label_col is not None:
            label = ws.cell(row=r, column=label_col, value=kind.capitalize())
            label.font = Font(bold=True)
