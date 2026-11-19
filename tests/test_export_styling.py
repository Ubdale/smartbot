"""Styling exports by asking: the model reads the request, code applies it."""

import io
from collections.abc import Iterator
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from openpyxl import load_workbook

from app.api.chat import STYLE_REQUEST
from app.api.deps import get_today
from app.exporting import build_answer_workbook
from app.llm.factory import get_llm
from app.pdf_report import build_answer_pdf
from app.query.executor import clear_cache
from app.styling import (
    ExportStyle,
    Highlight,
    SortSpec,
    apply_style,
    describe_style,
    row_matches,
    validate_style,
)
from tests.conftest import Account, signup
from tests.fakes import FakeLLM

SAMPLE = Path(__file__).resolve().parent.parent / "sample_data"
TODAY = date(2026, 11, 19)
TASKS: dict[str, Any] = {
    "columns": ["Task id", "Task", "Owner", "Due date", "Hours"],
    "rows": [
        [101, "Invoice run", "Sara", "2026-11-10", 4],
        [102, "Payroll", "Ali", "2026-11-25", 6],
        [103, "Audit prep", "Sara", None, 2],
        [104, "Board pack", "Omar", "2026-11-18", 8],
    ],
}


def _workbook(style: ExportStyle) -> object:
    data = build_answer_workbook(
        question="Open tasks", summary="4 tasks.", table=TASKS, chart=None,
        dataset_name="tasks.xlsx", created_at=datetime.now(UTC), style=style, today=TODAY,
    )  # fmt: skip
    return load_workbook(io.BytesIO(data))


def test_highlight_rules_are_decided_in_code() -> None:
    overdue = Highlight(column="Due date", op="before_today", color="red")
    assert row_matches(overdue, "2026-11-10", TODAY)
    assert not row_matches(overdue, "2026-11-25", TODAY)
    assert not row_matches(overdue, None, TODAY)  # no date is not overdue
    soon = Highlight(column="Due date", op="within_next_days", value=7)
    assert row_matches(soon, "2026-11-25", TODAY) and not row_matches(soon, "2026-12-25", TODAY)
    assert row_matches(Highlight(column="Hours", op=">", value=5), 6, TODAY)
    assert row_matches(Highlight(column="Owner", op="=", value="sara"), "Sara", TODAY)
    assert row_matches(Highlight(column="Task", op="contains", value="pay"), "Payroll", TODAY)
    assert row_matches(Highlight(column="Due date", op="empty"), None, TODAY)


def test_unknown_columns_and_colours_are_dropped_with_a_note() -> None:
    style = ExportStyle(
        header_color="sparkly",
        highlights=[Highlight(column="Priority", op="=", value="High"),
                    Highlight(column="due DATE", op="before_today", color="mauve")],
        sort=SortSpec(column="Nope"),
        hide_columns=["owner", "ghost"],
    )  # fmt: skip
    fixed, notes = validate_style(style, TASKS["columns"])
    assert fixed.header_color is None and fixed.sort is None
    assert [(h.column, h.color) for h in fixed.highlights] == [("Due date", "red")]
    assert fixed.hide_columns == ["Owner"]
    assert len(notes) == 4


def test_apply_style_sorts_hides_highlights_and_summarises() -> None:
    style = ExportStyle(
        highlights=[Highlight(column="Due date", op="before_today", color="red")],
        sort=SortSpec(column="Hours", descending=True),
        hide_columns=["Owner"],
        summary_rows=["average"],
    )
    t = apply_style(TASKS["columns"], TASKS["rows"], style, TODAY, numeric_columns=["Hours"])
    assert t.columns == ["Task id", "Task", "Due date", "Hours"]
    assert [r[0] for r in t.rows] == [104, 102, 101, 103]
    assert t.row_colors == ["red", None, "red", None]  # the hidden column still counts
    assert t.summary == [("Average", [None, None, None, 5.0])]
    assert "rows where Due date is before today in red" in describe_style(style)


def test_styled_workbook() -> None:
    style = ExportStyle(
        title="Overdue check",
        header_color="green",
        highlights=[Highlight(column="Due date", op="before_today", color="red")],
        summary_rows=["total", "average"],
        sort=SortSpec(column="Due date"),
    )
    wb = _workbook(style)
    ws = wb["Answer"]  # type: ignore[index]
    assert ws["A1"].value == "Overdue check"
    assert ws["A5"].fill.fgColor.rgb.endswith("16A34A")  # green header
    # Sorted by due date, blanks last: 101 (overdue), 104 (overdue), 102, 103.
    assert [ws.cell(row=r, column=1).value for r in range(6, 10)] == [101, 104, 102, 103]
    assert ws["B6"].fill.fgColor.rgb.endswith("FEE2E2")  # overdue rows in red
    assert ws["B7"].fill.fgColor.rgb.endswith("FEE2E2")
    assert not ws["B8"].fill.fgColor.rgb.endswith("FEE2E2")
    assert ws["E10"].value == "=SUM(E6:E9)"  # live formulas, not typed-in numbers
    assert ws["E11"].value == "=AVERAGE(E6:E9)"
    assert ws["A10"].value == "Total" and ws["A11"].value == "Average"  # IDs never totalled
    about = wb["How this was made"]  # type: ignore[index]
    assert any("green header" in str(c.value) for c in about["B"])


def test_styled_pdf() -> None:
    style = ExportStyle(
        header_color="#123456",
        highlights=[Highlight(column="Due date", op="before_today")],
        summary_rows=["max"],
        hide_columns=["Owner"],
    )
    pdf = build_answer_pdf(
        question="Open tasks", summary="4 tasks.", table=TASKS, chart=None,
        dataset_name="tasks.xlsx", created_at=datetime.now(UTC), style=style, today=TODAY,
    )  # fmt: skip
    assert pdf.startswith(b"%PDF")


def test_style_requests_are_recognised() -> None:
    for q in ["make the header green", "highlight overdue tasks in red", "add an average row",
              "hide the email column", "colour the rows by status"]:  # fmt: skip
        assert STYLE_REQUEST.search(q), q
    for q in ["sales by colour", "total sales by region", "show total rows per city"]:
        assert not STYLE_REQUEST.search(q), q


# --- through the chat -----------------------------------------------------------------


@pytest.fixture(autouse=True)
def _setup(app: FastAPI) -> Iterator[None]:
    clear_cache()
    app.dependency_overrides[get_today] = lambda: TODAY
    yield
    clear_cache()


def test_style_by_asking_builds_up_and_exports(app: FastAPI, client: TestClient) -> None:
    acct: Account = signup(client, "owner@acme.com", workspace="Acme")
    content = (SAMPLE / "sales_orders.csv").read_bytes()
    ds = client.post("/datasets", headers=acct.headers,
                     files={"file": ("sales_orders.csv", content)}).json()["id"]  # fmt: skip
    plan = {"metric": "sum", "metric_column": "revenue", "group_by": ["region"],
            "sort": {"by": "value", "direction": "desc"}, "chart": "bar"}  # fmt: skip
    first = {
        "header_color": "green",
        "summary_rows": ["average"],
        "highlights": [{"column": "region", "op": "=", "value": "North", "color": "yellow"}],
    }
    second = {**first, "sort": {"column": "Region", "descending": False}}  # code fixes the case
    llm = FakeLLM([plan, first, second], summary="North leads.")
    app.dependency_overrides[get_llm] = lambda: llm

    def ask(q: str, **ids: str) -> dict:  # type: ignore[type-arg]
        resp = client.post("/chat/ask", headers=acct.headers, json={"question": q, **ids})
        assert resp.status_code == 200, resp.text
        return resp.json()  # type: ignore[no-any-return]

    conv = ask("Revenue by region", dataset_id=ds)["conversation_id"]
    reply = ask("make the header green, highlight North in yellow and add an average row",
                conversation_id=conv)["reply"]  # fmt: skip
    assert reply["answer"]["type"] == "file"
    assert reply["answer"]["style"]["header_color"] == "green"
    assert "green header" in reply["content"]

    # A follow-up only makes sense after a file, and builds on the current settings.
    reply = ask("now sort it by region", conversation_id=conv)["reply"]
    assert '"header_color":"green"' in llm.plan_calls[-1][-1].content
    assert reply["answer"]["style"]["sort"] == {"column": "region"}

    resp = client.get(f"/chat/messages/{reply['id']}/export.xlsx", headers=acct.headers)
    assert resp.status_code == 200
    ws = load_workbook(io.BytesIO(resp.content))["Answer"]
    assert ws["A5"].fill.fgColor.rgb.endswith("16A34A")
    regions = [str(ws.cell(row=r, column=1).value) for r in range(6, 10)]
    assert regions == sorted(regions)
    north = 6 + regions.index("North")
    assert ws.cell(row=north, column=1).fill.fgColor.rgb.endswith("FEF9C3")
    assert ws["B10"].value == "=AVERAGE(B6:B9)" and ws["A10"].value == "Average"

    pdf = client.get(f"/chat/messages/{reply['id']}/export.pdf", headers=acct.headers)
    assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF")

    other = signup(client, "spy@other.com", workspace="Other")
    assert client.get(f"/chat/messages/{reply['id']}/export.xlsx",
                      headers=other.headers).status_code == 404  # fmt: skip
