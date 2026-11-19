"""A one-page-or-more PDF report of an answer: title, summary, bar chart, table."""

from datetime import datetime
from typing import Any

from fpdf import FPDF

from app.exporting import _pretty, answer_title, is_id_like

BRAND = (79, 70, 229)
BAND = (238, 242, 255)
INK = (24, 24, 27)
MUTED = (113, 113, 122)
MAX_ROWS = 200
MAX_BARS = 15

# The built-in PDF fonts only cover Latin-1: map common typography, replace the rest.
_ASCII = str.maketrans(
    {
        chr(0x201C): '"', chr(0x201D): '"', chr(0x2018): "'", chr(0x2019): "'",
        chr(0x2013): "-", chr(0x2014): "-", chr(0x2026): "...", chr(0x2192): "->",
        chr(0x2011): "-", chr(0x00A0): " ",
    }
)  # fmt: skip


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
) -> bytes:
    columns: list[str] = table.get("columns", [])
    rows: list[list[Any]] = table.get("rows", [])[:MAX_ROWS]
    explanation = explanation or {}
    title = answer_title(question, explanation.get("description", ""))
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
    pdf.set_font("Helvetica", size=10)
    pdf.set_text_color(63, 63, 70)
    pdf.ln(1)
    pdf.multi_cell(width, 5, _t(summary))
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(*MUTED)
    pdf.ln(1)
    dr = explanation.get("date_range")
    period = f" | {dr['label']}: {dr['start']} to {dr['end']}" if dr else ""
    pdf.multi_cell(width, 4, _t(f'From "{dataset_name}"{period} | {created_at:%d %b %Y %H:%M} UTC'))
    if explanation.get("data_note"):
        pdf.set_text_color(180, 83, 9)
        pdf.multi_cell(width, 4, _t(explanation["data_note"]))
    pdf.ln(4)

    # A simple horizontal bar chart for grouped answers.
    if chart and rows and chart.get("x") in columns and chart.get("y") in columns:
        xi, yi = columns.index(chart["x"]), columns.index(chart["y"])
        bars = [(r[xi], r[yi]) for r in rows if isinstance(r[yi], int | float)][:MAX_BARS]
        top = max((abs(float(v)) for _, v in bars), default=0.0)
        if bars and top > 0:
            label_w, value_w = width * 0.3, 28.0
            bar_area = width - label_w - value_w
            pdf.set_font("Helvetica", size=8)
            for label, value in bars:
                y = pdf.get_y()
                pdf.set_text_color(*INK)
                pdf.cell(label_w, 5, _t(label)[:40], align="R")
                pdf.set_fill_color(*BRAND)
                bar = max(bar_area * abs(float(value)) / top - 4, 0.5)
                pdf.rect(pdf.l_margin + label_w + 2, y + 1, bar, 3, style="F")
                pdf.set_x(pdf.l_margin + label_w + bar_area)
                pdf.cell(
                    value_w, 5, _fmt(value, chart["y"], decimals), new_x="LMARGIN", new_y="NEXT"
                )
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
            pdf.set_fill_color(*BRAND)
            pdf.set_text_color(255, 255, 255)
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
            pdf.set_fill_color(*(BAND if i % 2 else (255, 255, 255)))
            for j, c in enumerate(columns):
                text = _fmt(row[j], c, decimals)
                while text and pdf.get_string_width(text) > widths[j] - 2:
                    text = text[:-2] + "~"
                pdf.cell(widths[j], 6, text, fill=True, align="R" if numeric[j] else "L")
            pdf.ln()
        if len(table.get("rows", [])) > MAX_ROWS:
            pdf.set_text_color(*MUTED)
            pdf.cell(0, 6, f"Showing the first {MAX_ROWS} rows. Download Excel for all of them.")

    return bytes(pdf.output())
