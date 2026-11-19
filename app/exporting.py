"""Styled Excel workbooks for answers: title, summary, formatted table, native chart."""

import io
import re
from datetime import date, datetime
from typing import Any

from openpyxl import Workbook
from openpyxl.chart import BarChart, LineChart, PieChart, Reference
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet

BRAND = "4F46E5"  # indigo, as in the app
BAND = "EEF2FF"
GRID = Side(style="thin", color="E4E4E7")
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_ISO_DATETIME = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}")
_FORMULA = re.compile(r"^[=+\-@]")


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


def _number_format(values: list[Any]) -> str | None:
    nums = [v for v in values if isinstance(v, int | float) and not isinstance(v, bool)]
    if not nums or len(nums) < len([v for v in values if v is not None]):
        return None
    if all(float(v).is_integer() for v in nums):
        return "#,##0"
    return "#,##0.00"


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
) -> bytes:
    columns: list[str] = table.get("columns", [])
    rows: list[list[Any]] = table.get("rows", [])

    wb = Workbook()
    ws: Worksheet = wb.active  # type: ignore[assignment]
    ws.title = _sheet_name("Answer")
    ws.sheet_view.showGridLines = False

    # Title block
    ws["A1"] = question
    ws["A1"].font = Font(size=16, bold=True, color="18181B")
    ws["A2"] = summary
    ws["A2"].font = Font(size=11, color="3F3F46")
    ws["A2"].alignment = Alignment(wrap_text=True, vertical="top")
    ws["A3"] = f"From “{dataset_name}” · {created_at:%d %b %Y %H:%M} UTC · computed by AskData"
    ws["A3"].font = Font(size=9, italic=True, color="71717A")
    last_col = get_column_letter(max(len(columns), 1))
    ws.merge_cells(f"A2:{last_col}2")
    ws.row_dimensions[2].height = 15 * max(1, min(6, len(summary) // 90 + 1))

    # Table
    header_row = 5
    for j, name in enumerate(columns, start=1):
        cell = ws.cell(row=header_row, column=j, value=_pretty(name))
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor=BRAND)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = Border(bottom=GRID)
    ws.row_dimensions[header_row].height = 22

    formats = [_number_format([r[j] for r in rows]) for j in range(len(columns))]
    for i, row in enumerate(rows, start=header_row + 1):
        band = PatternFill("solid", fgColor=BAND) if (i - header_row) % 2 == 0 else None
        for j, value in enumerate(row, start=1):
            cell = ws.cell(row=i, column=j, value=_cell_value(value))
            cell.border = Border(bottom=GRID)
            if band:
                cell.fill = band
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
    ws.freeze_panes = f"A{header_row + 1}"

    # Column widths from the content (capped so long text wraps instead).
    for j, name in enumerate(columns, start=1):
        longest = max([len(_pretty(name))] + [len(str(r[j - 1])) for r in rows[:500]])
        ws.column_dimensions[get_column_letter(j)].width = min(max(10, longest + 3), 60)

    # A native Excel chart, editable like any other.
    if chart and rows and chart.get("x") in columns and chart.get("y") in columns:
        x, y = columns.index(chart["x"]) + 1, columns.index(chart["y"]) + 1
        kind = chart.get("type")
        excel_chart: BarChart | LineChart | PieChart
        if kind == "line":
            excel_chart = LineChart()
        elif kind == "pie":
            excel_chart = PieChart()
        else:
            excel_chart = BarChart()
            excel_chart.type = "col"
        excel_chart.title = question[:80]
        data = Reference(ws, min_col=y, min_row=header_row, max_row=last_row)
        cats = Reference(ws, min_col=x, min_row=header_row + 1, max_row=last_row)
        excel_chart.add_data(data, titles_from_data=True)
        excel_chart.set_categories(cats)
        excel_chart.height, excel_chart.width = 9, 18
        if not isinstance(excel_chart, PieChart):
            excel_chart.legend = None
        ws.add_chart(excel_chart, f"{get_column_letter(len(columns) + 2)}{header_row}")

    ws.print_title_rows = f"{header_row}:{header_row}"
    ws.page_setup.orientation = "landscape" if len(columns) > 4 else "portrait"
    ws.page_setup.fitToWidth = 1

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()
