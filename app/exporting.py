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

from app.styling import ExportStyle, apply_style, color, describe_style

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


_LEAD_NOISE = re.compile(
    r"^(if\s+i\s+(tell|ask)\s+you\s+to|can\s+you|could\s+you|would\s+you|will\s+you|please|pls|"
    r"kindly|i\s+want(\s+you)?\s+to|i\s+need|give\s+me|show\s+me|get\s+me|make(\s+me)?|"
    r"create|build|prepare|generate|export|download|list|an?|the|"
    r"(sheet|spreadsheet|excel|file|table|report|list|workbook)(\s+(which|that)(\s+will)?"
    r"\s+(have|has|show|shows|contain|contains|include|includes)|\s+(with|of|for))?)\s+",
    re.I,
)
_TAIL_NOISE = re.compile(
    r"\s+((would|will|can|could)\s+you(\s+(do|make|give)(\s+(it|that|this|me))?)?|please|pls"
    r"|thanks?(\s+you)?|for\s+me|in\s+(an?\s+)?(excel|sheet|pdf|file))$",
    re.I,
)


def _clean_question(question: str) -> str:
    """“if i tell you to make a sheet which will have task of Sara would you do it?”
    becomes “Task of Sara”: the request wording goes, the subject stays."""
    q = question.strip().rstrip("?.! ")
    for _ in range(8):
        shorter = _TAIL_NOISE.sub("", _LEAD_NOISE.sub("", q)).strip(" ,.?!")
        if shorter == q:
            break
        q = shorter
    return q[:1].upper() + q[1:]


def answer_title(question: str, description: str) -> str:
    """A clean title for exports: never the raw prompt. The question with the request
    wording removed when that reads well, else what was computed."""
    cleaned = _clean_question(question)
    if 3 <= len(cleaned) <= 80:
        return cleaned
    d = re.sub(r";\s*(showing up to|sorted by).*$", "", description).rstrip(". ")
    if d:
        return d[:120]
    if len(cleaned) >= 3:
        return cleaned[:77].rsplit(" ", 1)[0] + "…"
    return "AskData answer"


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
    title = answer_title(question, explanation.get("description", ""))
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
    elif (
        rows
        and metric in ("sum", "count", "count_distinct")
        and len(columns) > 1
        and (formats[-1] not in (None, "0"))
    ):
        letter = get_column_letter(value_col)
        rng = f"{letter}{header_row + 1}:{letter}{last_row}"
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

    _about_sheet(wb, question, title, explanation, dataset_name, created_at, len(rows),
                 describe_style(style) if style != ExportStyle() else None)  # fmt: skip
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
    formatting: str | None = None,
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
        *([("Formatting", formatting)] if formatting else []),
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
