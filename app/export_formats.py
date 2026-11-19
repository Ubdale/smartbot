"""Every download format for an answer. Table formats (Excel, PDF, CSV, Word, PowerPoint)
contain only the table; the chart exports on its own as a PNG.

All builders take the same arguments and apply the formatting asked for in chat
(ExportStyle) the same way, so a "green header, overdue in red" request looks the
same in every format. Charts are drawn by code from the answer's chart spec.
"""

import csv
import io
from collections.abc import Callable
from datetime import date, datetime
from typing import Any

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from pptx import Presentation
from pptx.dml.color import RGBColor as PptColor
from pptx.util import Inches as PptInches
from pptx.util import Pt as PptPt

from app.charts import render_chart_png
from app.exporting import (
    BAND,
    BRAND,
    _pretty,
    answer_title,
    build_answer_workbook,
    summable_columns,
)
from app.pdf_report import _fmt, build_answer_pdf
from app.styling import ExportStyle, StyledTable, apply_style, color

MAX_DOC_ROWS = 1000
SLIDE_ROWS = 12
MAX_SLIDES = 6


class NothingToExportError(ValueError):
    """The answer has nothing for this format (e.g. a chart PNG of an answer with no chart)."""


def _prepare(
    table: dict[str, Any], style: ExportStyle | None, today: date | None
) -> tuple[StyledTable, ExportStyle, frozenset[str]]:
    columns: list[str] = table.get("columns", [])
    rows: list[list[Any]] = table.get("rows", [])
    style = style or ExportStyle()
    styled = apply_style(columns, rows, style, today or date.today(),
                         numeric_columns=summable_columns(columns, rows))  # fmt: skip
    decimals = frozenset(
        c
        for j, c in enumerate(styled.columns)
        if any(isinstance(r[j], float) and not r[j].is_integer() for r in styled.rows)
    )
    return styled, style, decimals


def _show(value: Any, column: str, decimals: frozenset[str]) -> str:
    if value is None:
        return ""
    if isinstance(value, bool | int | float):
        return _fmt(value, column, decimals)
    return str(value)


def _summary_text(
    label: str, values: list[Any], columns: list[str], decimals: frozenset[str]
) -> list[str]:
    first = next((j for j, v in enumerate(values) if v is None), None)
    out = []
    for j, c in enumerate(columns):
        if j == first:
            out.append(label)
        elif values[j] is None:
            out.append("")
        else:
            out.append(_show(values[j], c, decimals | {c} if label == "Average" else decimals))
    return out


def _numeric(styled: StyledTable) -> list[bool]:
    return [
        any(isinstance(r[j], int | float) and not isinstance(r[j], bool) for r in styled.rows)
        for j in range(len(styled.columns))
    ]


# --- CSV ------------------------------------------------------------------------------


def build_answer_csv(
    *, table: dict[str, Any], style: ExportStyle | None = None, today: date | None = None,
    **_: Any,
) -> bytes:  # fmt: skip
    """Plain data (sorted/hidden as asked), with summary rows at the end. Opens in Excel."""
    styled, _style, _dec = _prepare(table, style, today)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(styled.columns)
    for row in styled.rows:
        w.writerow(["" if v is None else _safe_csv(v) for v in row])
    for label, values in styled.summary:
        first = next((j for j, v in enumerate(values) if v is None), None)
        w.writerow([label if j == first else ("" if v is None else round(v, 4))
                    for j, v in enumerate(values)])  # fmt: skip
    return buf.getvalue().encode("utf-8-sig")  # the BOM makes Excel read UTF-8


def _safe_csv(v: Any) -> Any:
    # Never let a cell from the data run as a formula when the CSV is opened in Excel.
    if isinstance(v, str) and v[:1] in ("=", "+", "-", "@") and not _is_number(v):
        return "'" + v
    return v


def _is_number(s: str) -> bool:
    try:
        float(s)
    except ValueError:
        return False
    return True


# --- Chart PNG ------------------------------------------------------------------------


def build_answer_png(
    *, question: str, table: dict[str, Any], chart: dict[str, Any] | None,
    explanation: dict[str, Any] | None = None, **_: Any,
) -> bytes:  # fmt: skip
    title = answer_title(question, (explanation or {}).get("description", ""))
    png = render_chart_png(
        chart, table.get("columns", []), table.get("rows", []), title=title,
        y_label=_pretty(chart["y"]) if chart else "", width=10, height=5.6, dpi=200,
    )  # fmt: skip
    if png is None:
        raise NothingToExportError("This answer has no chart to save as an image.")
    return png


# --- Word -----------------------------------------------------------------------------


def _shade(cell: Any, hex_: str) -> None:
    props = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_)
    props.append(shd)


def _cell_text(cell: Any, text: str, *, bold: bool = False, hex_: str | None = None,
               right: bool = False, size: float = 9) -> None:  # fmt: skip
    para = cell.paragraphs[0]
    para.alignment = WD_ALIGN_PARAGRAPH.RIGHT if right else WD_ALIGN_PARAGRAPH.LEFT
    run = para.add_run(text)
    run.font.size = Pt(size)
    run.bold = bold
    if hex_:
        run.font.color.rgb = RGBColor.from_string(hex_)


def build_answer_docx(
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
    """A Word file of the table: a title and the formatted table."""
    explanation = explanation or {}
    styled, style, decimals = _prepare(table, style, today)
    title = style.title or answer_title(question, explanation.get("description", ""))
    doc = Document()
    section = doc.sections[0]
    if len(styled.columns) > 4:
        section.orientation = WD_ORIENT.LANDSCAPE
        section.page_width, section.page_height = section.page_height, section.page_width
    for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(section, side, Inches(0.7))

    heading = doc.add_heading(title, level=1)
    for run in heading.runs:
        run.font.color.rgb = RGBColor.from_string("18181B")
    if styled.columns:
        numeric = _numeric(styled)
        rows = styled.rows[:MAX_DOC_ROWS]
        header_fill, header_text = styled.header or (BRAND, "FFFFFF")
        t = doc.add_table(rows=1, cols=len(styled.columns))
        t.style = "Table Grid"
        for j, c in enumerate(styled.columns):
            cell = t.rows[0].cells[j]
            _shade(cell, header_fill)
            _cell_text(cell, _pretty(c), bold=True, hex_=header_text, right=numeric[j])
        for i, row in enumerate(rows):
            cells = t.add_row().cells
            mark = color(styled.row_colors[i], strong=False)
            fill = mark[0] if mark else (BAND if style.banded_rows and i % 2 else None)
            for j, c in enumerate(styled.columns):
                if fill:
                    _shade(cells[j], fill)
                _cell_text(cells[j], _show(row[j], c, decimals), right=numeric[j],
                           hex_=mark[1] if mark else None)  # fmt: skip
        for label, values in styled.summary:
            cells = t.add_row().cells
            for j, text in enumerate(_summary_text(label, values, styled.columns, decimals)):
                _cell_text(cells[j], text, bold=True, right=numeric[j])
        if len(styled.rows) > MAX_DOC_ROWS:
            doc.add_paragraph(f"Showing the first {MAX_DOC_ROWS} rows. Download Excel for all.")
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


# --- PowerPoint -----------------------------------------------------------------------


def _textbox(slide: Any, text: str, left: float, top: float, width: float, height: float,
             *, size: int, bold: bool = False, hex_: str = "18181B") -> None:  # fmt: skip
    frame = slide.shapes.add_textbox(PptInches(left), PptInches(top), PptInches(width),
                                     PptInches(height)).text_frame  # fmt: skip
    frame.word_wrap = True
    run = frame.paragraphs[0].add_run()
    run.text = text
    run.font.size, run.font.bold = PptPt(size), bold
    run.font.color.rgb = PptColor.from_string(hex_)


def build_answer_pptx(
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
    """Slides of the table (12 rows per slide)."""
    explanation = explanation or {}
    styled, style, decimals = _prepare(table, style, today)
    title = style.title or answer_title(question, explanation.get("description", ""))
    prs = Presentation()
    prs.slide_width, prs.slide_height = PptInches(13.333), PptInches(7.5)
    blank = prs.slide_layouts[6]

    def new_slide(heading: str) -> Any:
        slide = prs.slides.add_slide(blank)
        bar = slide.shapes.add_shape(1, 0, 0, prs.slide_width, PptInches(0.12))
        bar.fill.solid()
        bar.fill.fore_color.rgb = PptColor.from_string(BRAND)
        bar.line.fill.background()
        _textbox(slide, heading, 0.6, 0.35, 12.1, 0.8, size=26, bold=True)
        return slide

    if styled.columns and styled.rows:
        numeric = _numeric(styled)
        header_fill, header_text = styled.header or (BRAND, "FFFFFF")
        chunks = [styled.rows[i : i + SLIDE_ROWS]
                  for i in range(0, len(styled.rows), SLIDE_ROWS)][:MAX_SLIDES]  # fmt: skip
        for n, chunk in enumerate(chunks):
            last = n == len(chunks) - 1
            extra = len(styled.summary) if last else 0
            s = new_slide(f"{title} ({n + 1}/{len(chunks)})" if len(chunks) > 1 else title)
            shape = s.shapes.add_table(1 + len(chunk) + extra, len(styled.columns),
                                       PptInches(0.6), PptInches(1.3), PptInches(12.1),
                                       PptInches(0.4 * (1 + len(chunk) + extra)))  # fmt: skip
            grid = shape.table

            def put(
                r: int, j: int, text: str, fill: str | None, ink: str, bold: bool, grid: Any = grid
            ) -> None:
                cell = grid.cell(r, j)
                cell.text = text
                para = cell.text_frame.paragraphs[0]
                para.alignment = 3 if numeric[j] else 1  # right / left
                font = para.runs[0].font if para.runs else para.font
                font.size, font.bold = PptPt(11), bold
                font.color.rgb = PptColor.from_string(ink)
                if fill:
                    cell.fill.solid()
                    cell.fill.fore_color.rgb = PptColor.from_string(fill)

            for j, c in enumerate(styled.columns):
                put(0, j, _pretty(c), header_fill, header_text, True)
            offset = n * SLIDE_ROWS
            for i, row in enumerate(chunk, start=1):
                mark = color(styled.row_colors[offset + i - 1], strong=False)
                fill = mark[0] if mark else (BAND if style.banded_rows and i % 2 == 0 else "FFFFFF")
                for j, c in enumerate(styled.columns):
                    put(
                        i, j, _show(row[j], c, decimals), fill, mark[1] if mark else "18181B", False
                    )
            if last:
                for k, (label, values) in enumerate(styled.summary):
                    texts = _summary_text(label, values, styled.columns, decimals)
                    for j, text in enumerate(texts):
                        put(1 + len(chunk) + k, j, text, "FFFFFF", "18181B", True)
        shown = SLIDE_ROWS * MAX_SLIDES
        if len(styled.rows) > shown:
            _textbox(s, f"Showing the first {shown} rows. Download Excel for all of them.",
                     0.6, 6.9, 12.1, 0.4, size=10, hex_="71717A")  # fmt: skip

    buf = io.BytesIO()
    prs.save(buf)
    return buf.getvalue()


# --- the registry ---------------------------------------------------------------------

Builder = Callable[..., bytes]
FORMATS: dict[str, tuple[Builder, str, str]] = {  # format: (builder, media type, name)
    "xlsx": (build_answer_workbook,
             "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "Excel"),
    "pdf": (build_answer_pdf, "application/pdf", "PDF"),
    "csv": (build_answer_csv, "text/csv; charset=utf-8", "CSV"),
    "docx": (build_answer_docx,
             "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "Word"),
    "pptx": (build_answer_pptx,
             "application/vnd.openxmlformats-officedocument.presentationml.presentation",
             "PowerPoint"),
    "png": (build_answer_png, "image/png", "chart image"),
}  # fmt: skip
