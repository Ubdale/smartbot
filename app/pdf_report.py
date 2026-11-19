"""A PDF of an answer's table: a title, the chart and the table (no response text)."""

import io
from datetime import date, datetime
from typing import Any

from fpdf import FPDF

from app.charts import render_chart_png
from app.exporting import _pretty, answer_title, is_id_like, summable_columns
from app.styling import ExportStyle, apply_style, color

BRAND = (79, 70, 229)
BAND = (238, 242, 255)
INK = (24, 24, 27)
MUTED = (113, 113, 122)
MAX_ROWS = 200

# The built-in PDF fonts only cover Latin-1: map common typography, replace the rest.
_ASCII = str.maketrans(
    {
        chr(0x201C): '"', chr(0x201D): '"', chr(0x2018): "'", chr(0x2019): "'",
        chr(0x2013): "-", chr(0x2014): "-", chr(0x2026): "...", chr(0x2192): "->",
        chr(0x2011): "-", chr(0x00A0): " ",
    }
)  # fmt: skip


def _rgb(hex_: str) -> tuple[int, int, int]:
    return (int(hex_[0:2], 16), int(hex_[2:4], 16), int(hex_[4:6], 16))


def _t(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.translate(_ASCII).encode("latin-1", "replace").decode("latin-1")


def _fmt(value: Any, column: str, decimals: frozenset[str] = frozenset()) -> str:
    """`decimals`: columns of this report with any fractional value (2 decimals for all)."""
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, int | float):
        if is_id_like(column) and float(value).is_integer():
            return str(int(value))
        # One style per column: 394,641.00 next to 392,208.95, not 394,641.
        return f"{value:,.2f}" if column in decimals else f"{value:,.0f}"
    return _t(value)


class _Report(FPDF):
    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("Helvetica", size=8)
        self.set_text_color(*MUTED)
        self.cell(0, 6, f"AskData - numbers computed from your data   |   page {self.page_no()}",
                  align="C")  # fmt: skip


def build_answer_pdf(
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
    all_rows: list[list[Any]] = table.get("rows", [])
    explanation = explanation or {}
    style = style or ExportStyle()
    styled = apply_style(
        columns, all_rows, style, today or date.today(),
        numeric_columns=summable_columns(columns, all_rows),
    )  # fmt: skip
    columns = styled.columns
    rows = styled.rows[:MAX_ROWS]
    title = style.title or answer_title(question, explanation.get("description", ""))
    header_fill, header_text = styled.header or ("4F46E5", "FFFFFF")
    decimals = frozenset(
        c
        for j, c in enumerate(columns)
        if any(isinstance(r[j], float) and not r[j].is_integer() for r in rows)
    )

    pdf = _Report(orientation="L" if len(columns) > 4 else "P", format="A4")
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.add_page()
    width = pdf.w - pdf.l_margin - pdf.r_margin

    # Header band
    pdf.set_fill_color(*BRAND)
    pdf.rect(0, 0, pdf.w, 4, style="F")
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(*INK)
    pdf.multi_cell(width, 8, _t(title))
    pdf.ln(4)

    # The chart (bar, line or pie), drawn by code from the original answer.
    png = render_chart_png(
        chart, table.get("columns", []), all_rows,
        y_label=_pretty(chart["y"]) if chart else "",
    )  # fmt: skip
    if png:
        chart_h = width * 4.6 / 9.0
        if pdf.get_y() + chart_h > pdf.h - 20:
            pdf.add_page()
        pdf.image(io.BytesIO(png), x=pdf.l_margin, w=width)
        pdf.ln(4)

    # The table: widths follow the content, long text is cut to fit.
    if columns:
        pdf.set_font("Helvetica", size=8)
        natural = [
            max([pdf.get_string_width(_t(_pretty(c))) + 4]
                + [pdf.get_string_width(_fmt(r[j], c, decimals)) + 4 for r in rows[:100]])
            for j, c in enumerate(columns)
        ]  # fmt: skip
        scale = min(1.0, width / sum(natural))
        widths = [w * scale for w in natural]
        numeric = [any(isinstance(r[j], int | float) for r in rows) for j in range(len(columns))]

        def header() -> None:
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_fill_color(*_rgb(header_fill))
            pdf.set_text_color(*_rgb(header_text))
            for j, c in enumerate(columns):
                pdf.cell(widths[j], 7, _t(_pretty(c)), fill=True, align="R" if numeric[j] else "L")
            pdf.ln()
            pdf.set_font("Helvetica", size=8)
            pdf.set_text_color(*INK)

        header()
        for i, row in enumerate(rows):
            if pdf.get_y() > pdf.h - 24:
                pdf.add_page()
                header()
            mark = color(styled.row_colors[i], strong=False)
            if mark:
                pdf.set_fill_color(*_rgb(mark[0]))
                pdf.set_text_color(*_rgb(mark[1]))
            else:
                band = style.banded_rows and i % 2
                pdf.set_fill_color(*(BAND if band else (255, 255, 255)))
                pdf.set_text_color(*INK)
            for j, c in enumerate(columns):
                text = _fmt(row[j], c, decimals)
                while text and pdf.get_string_width(text) > widths[j] - 2:
                    text = text[:-2] + "~"
                pdf.cell(widths[j], 6, text, fill=True, align="R" if numeric[j] else "L")
            pdf.ln()
        # Summary rows (computed by code over all rows, not just the ones shown).
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(*INK)
        pdf.set_fill_color(255, 255, 255)
        for label, values in styled.summary:
            first_text = next((j for j, v in enumerate(values) if v is None), None)
            for j, c in enumerate(columns):
                v = values[j]
                if j == first_text:
                    text = label
                elif v is None:
                    text = ""
                else:
                    text = _fmt(v, c, decimals | {c} if label == "Average" else decimals)
                pdf.cell(widths[j], 6, _t(text), border="T", align="R" if numeric[j] else "L")
            pdf.ln()
        if len(styled.rows) > MAX_ROWS:
            pdf.set_text_color(*MUTED)
            pdf.cell(0, 6, f"Showing the first {MAX_ROWS} rows. Download Excel for all of them.")

    return bytes(pdf.output())
