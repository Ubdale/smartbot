"""Every download format: asked for in chat, built by code, with charts and formatting."""

import csv
import io
from collections.abc import Iterator
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest
from docx import Document
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pptx import Presentation

from app.api.chat import requested_format
from app.api.deps import get_today
from app.charts import chart_series, render_chart_png
from app.export_formats import (
    FORMATS,
    NothingToExportError,
    build_answer_csv,
    build_answer_docx,
    build_answer_png,
    build_answer_pptx,
)
from app.llm.factory import get_llm
from app.query.executor import clear_cache
from app.styling import ExportStyle, Highlight, SortSpec
from tests.conftest import Account, signup
from tests.fakes import FakeLLM

SAMPLE = Path(__file__).resolve().parent.parent / "sample_data"
TODAY = date(2026, 11, 19)
PNG = b"\x89PNG\r\n\x1a\n"
TABLE: dict[str, Any] = {
    "columns": ["region", "sum_revenue"],
    "rows": [["North", 1500.5], ["South", 250.25], ["East", 900.0], ["West", None]],
}
BAR = {"type": "bar", "x": "region", "y": "sum_revenue"}
COMMON: dict[str, Any] = {
    "question": "Revenue by region", "summary": "North leads.", "table": TABLE, "chart": BAR,
    "dataset_name": "sales.csv", "created_at": datetime.now(UTC), "today": TODAY,
}  # fmt: skip


def test_requested_format() -> None:
    cases = {
        "export to pdf": "pdf",
        "give me this as a PDF with graphs": "pdf",
        "make it a powerpoint": "pptx",
        "turn this into slides": "pptx",
        "export it as a word document": "docx",
        "save it in word": "docx",
        "download csv": "csv",
        "give me formatted excel": "xlsx",
        "download it": "xlsx",
        "save the chart as an image": "png",
        "download the graph": "png",
        "download the excel with the chart": "xlsx",
        # Data questions stay data questions.
        "show me a chart of sales by month": None,
        "which slides do customers buy most": None,
        "revenue by region": None,
        "how many words in the title column": None,
    }
    for question, fmt in cases.items():
        assert requested_format(question) == fmt, question


def test_chart_series_and_png() -> None:
    cats, series = chart_series(BAR, TABLE["columns"], TABLE["rows"]) or ([], {})
    assert cats == ["North", "South", "East"]  # the row with no value is left out
    assert series == {"sum_revenue": [1500.5, 250.25, 900.0]}
    for kind in ("bar", "line", "pie"):
        png = render_chart_png({**BAR, "type": kind}, TABLE["columns"], TABLE["rows"])
        assert png is not None and png.startswith(PNG)
    assert render_chart_png(None, TABLE["columns"], TABLE["rows"]) is None
    multi = chart_series(
        {"type": "line", "x": "m", "y": "v", "series": "s"},
        ["m", "s", "v"], [["Jan", "A", 1], ["Jan", "B", 2], ["Feb", "A", 3]],
    )  # fmt: skip
    assert multi == (["Jan", "Feb"], {"A": [1.0, 3.0], "B": [2.0, None]})


def test_png_needs_a_chart() -> None:
    assert build_answer_png(**COMMON).startswith(PNG)
    with pytest.raises(NothingToExportError):
        build_answer_png(**{**COMMON, "chart": None})


def test_csv_follows_the_formatting() -> None:
    style = ExportStyle(sort=SortSpec(column="sum_revenue", descending=True),
                        summary_rows=["total"])  # fmt: skip
    text = build_answer_csv(**COMMON, style=style).decode("utf-8-sig")
    rows = list(csv.reader(io.StringIO(text)))
    assert rows[0] == ["region", "sum_revenue"]
    assert [r[0] for r in rows[1:5]] == ["North", "East", "South", "West"]
    assert rows[5] == ["Total", "2650.75"]
    evil = build_answer_csv(table={"columns": ["a"], "rows": [["=cmd()"], ["-5"]]})
    assert "'=cmd()" in evil.decode("utf-8-sig") and "'-5" not in evil.decode("utf-8-sig")


def test_word_file_has_only_the_formatted_table() -> None:
    style = ExportStyle(
        header_color="green",
        highlights=[Highlight(column="region", op="=", value="North", color="yellow")],
    )
    doc = Document(io.BytesIO(build_answer_docx(**COMMON, style=style)))
    assert doc.paragraphs[0].text == "Revenue by region"
    assert all("North leads" not in p.text for p in doc.paragraphs)  # no response text
    assert len(doc.inline_shapes) == 0  # no chart: charts export as PNG
    table = doc.tables[0]
    assert [c.text for c in table.rows[0].cells] == ["region", "Total revenue"]
    assert table.rows[1].cells[0].text == "North"
    xml = table.rows[0].cells[0]._tc.xml
    assert 'w:fill="16A34A"' in xml  # green header
    assert 'w:fill="FEF9C3"' in table.rows[1].cells[0]._tc.xml  # North in yellow


def test_powerpoint_has_only_table_slides() -> None:
    rows = [[f"Region {i}", float(i)] for i in range(30)]
    deck = Presentation(io.BytesIO(build_answer_pptx(
        **{**COMMON, "table": {"columns": TABLE["columns"], "rows": rows}},
        style=ExportStyle(summary_rows=["average"]),
    )))  # fmt: skip
    slides = list(deck.slides)
    assert len(slides) == 3  # 30 rows over 3 slides of 12, no chart slide
    assert not any(shape.has_chart for s in slides for shape in s.shapes)
    last_table = next(s for s in slides[-1].shapes if s.has_table).table
    assert last_table.cell(len(last_table.rows) - 1, 0).text == "Average"
    assert last_table.cell(len(last_table.rows) - 1, 1).text == "14.50"


def test_every_format_builds_without_a_chart() -> None:
    for fmt, (builder, _media, _name) in FORMATS.items():
        if fmt == "png":
            continue
        assert builder(**{**COMMON, "chart": None}), fmt


# --- through the chat -----------------------------------------------------------------


@pytest.fixture(autouse=True)
def _setup(app: FastAPI) -> Iterator[None]:
    clear_cache()
    app.dependency_overrides[get_today] = lambda: TODAY
    yield
    clear_cache()


def test_ask_for_any_format_in_chat(app: FastAPI, client: TestClient) -> None:
    acct: Account = signup(client, "owner@acme.com", workspace="Acme")
    content = (SAMPLE / "sales_orders.csv").read_bytes()
    ds = client.post("/datasets", headers=acct.headers,
                     files={"file": ("sales_orders.csv", content)}).json()["id"]  # fmt: skip
    plan = {"metric": "sum", "metric_column": "revenue", "group_by": ["region"],
            "sort": {"by": "value", "direction": "desc"}, "chart": "bar"}  # fmt: skip
    llm = FakeLLM([plan, {"header_color": "green"}], summary="North leads.")
    app.dependency_overrides[get_llm] = lambda: llm

    def ask(q: str, **ids: str) -> dict[str, Any]:
        resp = client.post("/chat/ask", headers=acct.headers, json={"question": q, **ids})
        assert resp.status_code == 200, resp.text
        reply: dict[str, Any] = resp.json()["reply"]
        return {**reply, "conversation_id": resp.json()["conversation_id"]}

    conv = ask("Revenue by region", dataset_id=ds)["conversation_id"]
    calls = len(llm.calls)
    expected = {
        "export this to pdf with the graph": ("pdf", b"%PDF"),
        "now make it a powerpoint": ("pptx", b"PK"),
        "export it as a word document": ("docx", b"PK"),
        "download csv": ("csv", b"\xef\xbb\xbf"),
        "save the chart as an image": ("png", PNG),
        "give me excel": ("xlsx", b"PK"),
    }
    for question, (fmt, magic) in expected.items():
        reply = ask(question, conversation_id=conv)
        assert reply["answer"]["format"] == fmt, question
        resp = client.get(f"/chat/messages/{reply['id']}/export.{fmt}", headers=acct.headers)
        assert resp.status_code == 200, (question, resp.text)
        assert resp.content.startswith(magic), question
        assert f'.{fmt}"' in resp.headers["content-disposition"]
    assert len(llm.calls) == calls  # none of these needed the model

    # Formatting and a format in one go; the formatting carries over to other formats.
    reply = ask("make the header green and export as pdf", conversation_id=conv)
    assert reply["answer"]["format"] == "pdf"
    assert reply["answer"]["style"] == {"header_color": "green"}
    later = ask("now give it to me in word", conversation_id=conv)
    assert later["answer"]["format"] == "docx"
    assert later["answer"]["style"] == {"header_color": "green"}

    assert client.get(f"/chat/messages/{reply['id']}/export.exe",
                      headers=acct.headers).status_code == 404  # fmt: skip
