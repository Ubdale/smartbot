"""Styled Excel workbooks for answers: title, summary, formatted table, native chart."""

import io
import re
from datetime import date, datetime
from typing import Any

from openpyxl import Workbook
from openpyxl.chart import BarChart, LineChart, PieChart, Reference
from openpyxl.formatting.rule import DataBarRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet

BRAND = "4F46E5"  # indigo, as in the app
BAND = "EEF2FF"
GRID = Side(style="thin", color="E4E4E7")
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_ISO_DATETIME = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}")
_FORMULA = re.compile(r"^[=+\-@]")
# Identifiers are numbers you never add up: no thousands separators for them.
_ID_LIKE = re.compile(r"(^|[\s_\-])(id|key|code|no|number|num|zip|phone|year|ref)($|[\s_\-])", re.I)
_TITLE_NOISE = re.compile(
    r"^(if i (tell|ask) you to|can you|could you|would you|please|pls|i want( you)? to|"
    r"give me|show me|make( me)?|create)\b",
    re.I,
)


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


def answer_title(question: str, description: str) -> str:
    """A clean title: the question when it reads like one, else what was computed."""
    q = question.strip().rstrip("?.! ")
    if len(q) <= 70 and not _TITLE_NOISE.search(q) and " would you " not in f" {q.lower()} ":
        return q[:1].upper() + q[1:]
    d = re.sub(r";\s*(showing up to|sorted by).*$", "", description).rstrip(". ")
    return (d or q)[:120]


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
) -> bytes:
    columns: list[str] = table.get("columns", [])
    rows: list[list[Any]] = table.get("rows", [])
    explanation = explanation or {}
    title = answer_title(question, explanation.get("description", ""))
    plan = explanation.get("plan") or {}

    wb = Workbook()
    ws: Worksheet = wb.active  # type: ignore[assignment]
    ws.title = _sheet_name("Answer")
    ws.sheet_view.showGridLines = False

    # Title block
    ws["A1"] = title
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

    formats = [_number_format([r[j] for r in rows], columns[j]) for j in range(len(columns))]
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

    # Grouped totals/counts: a live Totals row (Excel formulas) and data bars.
    metric = plan.get("metric")
    value_col = len(columns)
    if (
        rows
        and metric in ("sum", "count", "count_distinct")
        and len(columns) > 1
        and (formats[-1] not in (None, "0"))
    ):
        letter = get_column_letter(value_col)
        rng = f"{letter}{header_row + 1}:{letter}{last_row}"
        ws.conditional_formatting.add(
            rng,
            DataBarRule(start_type="min", end_type="max", color="A5B4FC"),  # type: ignore[no-untyped-call]
        )
        if metric == "sum" or (metric == "count" and not plan.get("time_grain")):
            total_row = last_row + 1
            ws.cell(row=total_row, column=1, value="Total").font = Font(bold=True)
            total = ws.cell(row=total_row, column=value_col, value=f"=SUM({rng})")
            total.font = Font(bold=True)
            total.number_format = formats[-1] or "#,##0"
            for j in range(1, value_col + 1):
                ws.cell(row=total_row, column=j).border = Border(
                    top=Side(style="medium", color=BRAND)
                )
    ws.freeze_panes = f"A{header_row + 1}"

    # Column widths from the content (capped so long text wraps instead).
    for j, name in enumerate(columns, start=1):
        # +5 leaves room for the filter arrow next to the header text.
        longest = max([len(_pretty(name)) + 5] + [len(str(r[j - 1])) + 2 for r in rows[:500]])
        ws.column_dimensions[get_column_letter(j)].width = min(max(10, longest), 60)

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
        excel_chart.title = title[:80]
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

    _about_sheet(wb, question, title, explanation, dataset_name, created_at, len(rows))
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _about_sheet(
    wb: Workbook,
    question: str,
    title: str,
    explanation: dict[str, Any],
    dataset_name: str,
    created_at: datetime,
    row_count: int,
) -> None:
    """How the numbers were made, for anyone the file is forwarded to."""
    ws = wb.create_sheet("How this was made")
    ws.sheet_view.showGridLines = False
    dr = explanation.get("date_range")
    lines: list[tuple[str, Any]] = [
        ("Title", title),
        ("Question asked", question),
        ("What was computed", explanation.get("description", "")),
        ("Time period", f"{dr['label']}: {dr['start']} to {dr['end']}" if dr else "All dates"),
        ("Source file", dataset_name),
        ("Rows in this answer", row_count),
        ("Created", f"{created_at:%d %b %Y %H:%M} UTC"),
        ("Note", explanation.get("data_note") or
         "Every number was computed from your data by AskData's database engine, not "
         "written by AI."),
    ]  # fmt: skip
    ws.column_dimensions["A"].width = 22
    ws.column_dimensions["B"].width = 90
    for i, (label, value) in enumerate(lines, start=1):
        ws.cell(row=i, column=1, value=label).font = Font(bold=True, color=BRAND)
        cell = ws.cell(row=i, column=2, value=_cell_value(value))
        cell.alignment = Alignment(wrap_text=True, vertical="top")
