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
