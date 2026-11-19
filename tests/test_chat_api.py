"""The ask flow end to end, with a scripted fake LLM."""

import re
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from app.api.deps import get_today
from app.llm.factory import get_llm
from app.models import UsageEvent
from app.query.executor import clear_cache
from tests.conftest import Account, signup
from tests.fakes import FakeLLM

SAMPLE = Path(__file__).resolve().parent.parent / "sample_data"
TODAY = date(2026, 11, 19)
REVENUE_BY_REGION = {
    "metric": "sum",
    "metric_column": "revenue",
    "group_by": ["region"],
    "sort": {"by": "value", "direction": "desc"},
    "chart": "bar",
}


@pytest.fixture(autouse=True)
def _setup(app: FastAPI) -> Iterator[None]:
    clear_cache()
    app.dependency_overrides[get_today] = lambda: TODAY
    yield
    clear_cache()


@pytest.fixture
def acct(client: TestClient) -> Account:
    return signup(client, "owner@acme.com", workspace="Acme")


@pytest.fixture
def dataset_id(client: TestClient, acct: Account) -> str:
    content = (SAMPLE / "sales_orders.csv").read_bytes()
    resp = client.post(
        "/datasets", headers=acct.headers, files={"file": ("sales_orders.csv", content)}
    )
    assert resp.status_code == 201, resp.text
    return str(resp.json()["id"])


def use_llm(app: FastAPI, llm: FakeLLM) -> FakeLLM:
    app.dependency_overrides[get_llm] = lambda: llm
    return llm


def ask(client: TestClient, acct: Account, question: str, **ids: str) -> Any:
    return client.post("/chat/ask", headers=acct.headers, json={"question": question, **ids})


def first_row_summary(prompt: str) -> str:
    """A well-behaved fake: quotes the top row of the table it was given."""
    rows = [line for line in prompt.split("Result:\n", 1)[1].splitlines()[1:] if "|" in line]
    label, value = (part.strip() for part in rows[0].split("|"))
    return f"{label} leads with {value} in revenue."


# --- happy path ---------------------------------------------------------------------


def test_ask_returns_computed_table_chart_summary_and_explanation(
    app: FastAPI, client: TestClient, acct: Account, dataset_id: str
) -> None:
    llm = use_llm(app, FakeLLM([REVENUE_BY_REGION], summary=first_row_summary))
    resp = ask(client, acct, "Which region brings in the most revenue?", dataset_id=dataset_id)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    reply = body["reply"]
    assert reply["status"] == "answered"

    expected = (
        pd.read_csv(SAMPLE / "sales_orders.csv")
        .groupby("region")["revenue"]
        .sum()
        .sort_values(ascending=False)
    )
    table = reply["answer"]["table"]
    assert table["columns"] == ["region", "sum_revenue"]
    assert [r[0] for r in table["rows"]] == list(expected.index)
    assert table["rows"][0][1] == pytest.approx(expected.iloc[0])

    assert reply["answer"]["chart"] == {
        "type": "bar", "x": "region", "y": "sum_revenue", "series": None
    }  # fmt: skip
    assert reply["content"].startswith(f"{expected.index[0]} leads with")
    assert reply["answer"]["summary_source"] == "llm"

    exp = reply["answer"]["explanation"]
    assert exp["description"].startswith("Total revenue, by region")
    assert "read_parquet(?)" in exp["sql"]
    assert exp["plan"]["metric_column"] == "revenue"

    # The model only ever saw the column summary: no row beyond the 3 samples per column.
    prompt = llm.plan_calls[0][-1].content
    assert "SO-00456" in prompt  # a sample value
    assert "SO-02999" not in prompt and "SO-01500" not in prompt


def test_summary_with_invented_number_is_replaced(
    app: FastAPI, client: TestClient, acct: Account, dataset_id: str
) -> None:
    use_llm(app, FakeLLM([REVENUE_BY_REGION], summary="Total revenue was 9,999,999 overall."))
    reply = ask(client, acct, "Revenue by region", dataset_id=dataset_id).json()["reply"]
    assert "9,999,999" not in reply["content"]
    assert reply["answer"]["summary_source"] == "template"
    top = reply["answer"]["table"]["rows"][0]
    assert reply["content"].startswith(f"Total revenue was highest for {top[0]} (")


def test_dates_are_resolved_in_code(
    app: FastAPI, client: TestClient, acct: Account, dataset_id: str
) -> None:
    use_llm(app, FakeLLM([{"metric": "count", "date_phrase": "last quarter"}], summary="Done."))
    reply = ask(client, acct, "How many orders last quarter?", dataset_id=dataset_id).json()[
        "reply"
    ]
    df = pd.read_csv(SAMPLE / "sales_orders.csv", parse_dates=["order_date"])
    expected = ((df.order_date >= "2026-07-01") & (df.order_date < "2026-10-01")).sum()
    assert reply["answer"]["table"]["rows"] == [[int(expected)]]
    assert reply["answer"]["explanation"]["date_range"] == {
        "start": "2026-07-01", "end": "2026-09-30", "label": "last quarter"
    }  # fmt: skip


# --- follow-ups, retries, clarification, failures -----------------------------------


def test_follow_up_sends_previous_plan(
    app: FastAPI, client: TestClient, acct: Account, dataset_id: str
) -> None:
    split = {**REVENUE_BY_REGION, "group_by": ["region", "city"], "limit": 50}
    llm = use_llm(app, FakeLLM([REVENUE_BY_REGION, split], summary="Here you go."))
    first = ask(client, acct, "Revenue by region", dataset_id=dataset_id).json()
    second = ask(client, acct, "now split by city", conversation_id=first["conversation_id"])
    assert second.status_code == 200, second.text
    assert second.json()["conversation_id"] == first["conversation_id"]

    follow_up_prompt = llm.plan_calls[1]
    history = [m.content for m in follow_up_prompt[-3:-1]]
    assert history[0] == "Question: Revenue by region"
    assert '"group_by":["region"]' in history[1]
    assert second.json()["reply"]["answer"]["table"]["columns"] == ["region", "city", "sum_revenue"]


def test_invalid_plan_is_retried_once_with_the_error(
    app: FastAPI, client: TestClient, acct: Account, dataset_id: str,
    session_factory: sessionmaker[Session],
) -> None:  # fmt: skip
    bad = {"metric": "sum", "metric_column": "sales"}
    good = {"metric": "sum", "metric_column": "revenue"}
    llm = use_llm(app, FakeLLM([bad, good], summary="Done."))
    reply = ask(client, acct, "total sales", dataset_id=dataset_id).json()["reply"]
    assert reply["status"] == "answered"
    retry_msg = llm.plan_calls[1][-1].content
    assert "Unknown column 'sales'" in retry_msg
    with session_factory() as db:
        purposes = [u.purpose for u in db.scalars(select(UsageEvent))]
    assert sorted(purposes) == ["plan", "plan", "summary"]


def test_two_bad_plans_ask_user_to_rephrase(
    app: FastAPI, client: TestClient, acct: Account, dataset_id: str
) -> None:
    use_llm(app, FakeLLM(["not json at all", {"metric": "sum", "metric_column": "profit"}]))
    reply = ask(client, acct, "profit margin?", dataset_id=dataset_id).json()["reply"]
    assert reply["status"] == "failed"
    assert reply["answer"] is None
    assert "rephrase" in reply["content"]
    assert "Unknown column 'profit'" in reply["content"]


def test_clarification(app: FastAPI, client: TestClient, acct: Account, dataset_id: str) -> None:
    use_llm(app, FakeLLM([{"clarification_needed": "Do you mean revenue or quantity?"}]))
    reply = ask(client, acct, "How big were sales?", dataset_id=dataset_id).json()["reply"]
    assert reply["status"] == "clarification"
    assert reply["content"] == "Do you mean revenue or quantity?"
    assert reply["answer"] is None


def test_no_rows_gets_template_summary_without_llm(
    app: FastAPI, client: TestClient, acct: Account, dataset_id: str
) -> None:
    plan = {"metric": "count", "group_by": ["region"],
            "filters": [{"column": "region", "op": "=", "value": "Antarctica"}]}  # fmt: skip
    llm = use_llm(app, FakeLLM([plan]))
    reply = ask(client, acct, "orders in Antarctica", dataset_id=dataset_id).json()["reply"]
    assert reply["answer"]["table"]["rows"] == []
    assert reply["content"].startswith("No rows matched")
    assert len(llm.calls) == 1  # no summary call


def test_llm_down_is_503(app: FastAPI, client: TestClient, acct: Account, dataset_id: str) -> None:
    use_llm(app, FakeLLM(unavailable=True))
    resp = ask(client, acct, "anything", dataset_id=dataset_id)
    assert resp.status_code == 503
    assert "switched off" in resp.json()["detail"]


def test_repeat_question_is_served_from_cache(
    app: FastAPI, client: TestClient, acct: Account, dataset_id: str
) -> None:
    use_llm(app, FakeLLM([REVENUE_BY_REGION, REVENUE_BY_REGION], summary="Done."))
    a = ask(client, acct, "Revenue by region", dataset_id=dataset_id).json()["reply"]
    b = ask(client, acct, "Revenue by region", dataset_id=dataset_id).json()["reply"]
    assert not a["answer"]["explanation"]["cached"]
    assert b["answer"]["explanation"]["cached"]
    assert a["answer"]["table"] == b["answer"]["table"]


def test_request_validation(client: TestClient, acct: Account, dataset_id: str) -> None:
    assert ask(client, acct, "   ", dataset_id=dataset_id).status_code == 422
    assert ask(client, acct, "x" * 501, dataset_id=dataset_id).status_code == 422
    assert ask(client, acct, "no target").status_code == 422


# --- conversations + isolation -------------------------------------------------------


def test_conversation_history_and_delete(
    app: FastAPI, client: TestClient, acct: Account, dataset_id: str
) -> None:
    use_llm(app, FakeLLM([REVENUE_BY_REGION, {"metric": "count"}], summary="Done."))
    conv_id = ask(client, acct, "Revenue by region", dataset_id=dataset_id).json()[
        "conversation_id"
    ]
    ask(client, acct, "and how many orders?", conversation_id=conv_id)

    listed = client.get("/chat/conversations", headers=acct.headers).json()
    assert [c["id"] for c in listed] == [conv_id]
    assert listed[0]["title"] == "Revenue by region"

    conv = client.get(f"/chat/conversations/{conv_id}", headers=acct.headers).json()
    assert [(m["role"], m["status"]) for m in conv["messages"]] == [
        ("user", None), ("assistant", "answered"), ("user", None), ("assistant", "answered")
    ]  # fmt: skip
    assert conv["messages"][1]["answer"]["table"]["columns"] == ["region", "sum_revenue"]

    assert client.delete(f"/chat/conversations/{conv_id}", headers=acct.headers).status_code == 204
    assert client.get(f"/chat/conversations/{conv_id}", headers=acct.headers).status_code == 404


def test_other_workspace_cannot_use_dataset_or_conversation(
    app: FastAPI, client: TestClient, acct: Account, dataset_id: str
) -> None:
    use_llm(app, FakeLLM([REVENUE_BY_REGION], summary="Done."))
    conv_id = ask(client, acct, "Revenue by region", dataset_id=dataset_id).json()[
        "conversation_id"
    ]
    other = signup(client, "spy@other.com", workspace="Other")

    assert ask(client, other, "Revenue?", dataset_id=dataset_id).status_code == 404
    assert ask(client, other, "Revenue?", conversation_id=conv_id).status_code == 404
    assert client.get(f"/chat/conversations/{conv_id}", headers=other.headers).status_code == 404
    assert client.get("/chat/conversations", headers=other.headers).json() == []


def test_teammate_cannot_read_private_conversation(
    app: FastAPI, client: TestClient, acct: Account, dataset_id: str,
    session_factory: sessionmaker[Session],
) -> None:  # fmt: skip
    """Same workspace, different user: conversations are private."""
    from app.core.security import hash_password
    from app.models import User

    with session_factory() as db:
        db.add(User(email="mate@acme.com", password_hash=hash_password("teammate-pass"),
                    workspace_id=acct.workspace["id"]))  # fmt: skip
        db.commit()
    token = client.post(
        "/auth/login", data={"username": "mate@acme.com", "password": "teammate-pass"}
    ).json()["access_token"]
    mate = {"Authorization": f"Bearer {token}"}

    use_llm(app, FakeLLM([REVENUE_BY_REGION], summary="Done."))
    conv_id = ask(client, acct, "Revenue by region", dataset_id=dataset_id).json()[
        "conversation_id"
    ]
    assert client.get(f"/chat/conversations/{conv_id}", headers=mate).status_code == 404
    # ...but the teammate can ask about the shared dataset in their own conversation.
    assert client.get(f"/datasets/{dataset_id}", headers=mate).status_code == 200


def test_usage_is_logged_per_workspace(
    app: FastAPI, client: TestClient, acct: Account, dataset_id: str,
    session_factory: sessionmaker[Session],
) -> None:  # fmt: skip
    use_llm(app, FakeLLM([REVENUE_BY_REGION], summary="Done."))
    ask(client, acct, "Revenue by region", dataset_id=dataset_id)
    with session_factory() as db:
        rows = list(db.scalars(select(UsageEvent)))
        total = db.scalar(select(func.sum(UsageEvent.prompt_tokens)))
    assert {u.workspace_id for u in rows} == {acct.workspace["id"]}
    assert {(u.provider, u.model) for u in rows} == {("fake", "fake-1")}
    assert total == 200


def test_sql_shown_to_user_has_placeholders_only(
    app: FastAPI, client: TestClient, acct: Account, dataset_id: str
) -> None:
    plan = {"metric": "count", "filters": [{"column": "customer", "op": "=", "value": "Apex Labs"}]}
    use_llm(app, FakeLLM([plan], summary="Done."))
    exp = ask(client, acct, "Apex orders", dataset_id=dataset_id).json()["reply"]["answer"][
        "explanation"
    ]
    assert "Apex Labs" not in exp["sql"]
    assert exp["params"] == ["Apex Labs"]
    assert not re.search(r"[A-Za-z]:[\\/]|/tmp/|data/workspaces", exp["sql"] + str(exp["params"]))


# --- listing rows and describing the dataset ----------------------------------------


def test_list_rows_shows_the_actual_rows(
    app: FastAPI, client: TestClient, acct: Account, dataset_id: str
) -> None:
    plan = {
        "metric": "rows",
        "columns": ["order_id", "customer", "revenue"],
        "filters": [{"column": "product", "op": "=", "value": "Training Day"},
                    {"column": "quantity", "op": "=", "value": 3}],
        "sort": {"by": "revenue", "direction": "desc"},
        "limit": 5,
    }  # fmt: skip
    use_llm(app, FakeLLM([plan], summary="Here they are."))
    reply = ask(client, acct, "show the biggest 3-day training orders", dataset_id=dataset_id)
    answer = reply.json()["reply"]["answer"]
    df = pd.read_csv(SAMPLE / "sales_orders.csv")
    expected = df[(df["product"] == "Training Day") & (df.quantity == 3)].sort_values(
        "revenue", ascending=False
    )
    assert answer["table"]["columns"] == ["order_id", "customer", "revenue"]
    assert [r[2] for r in answer["table"]["rows"]] == pytest.approx(list(expected.revenue[:5]))
    assert {r[0] for r in answer["table"]["rows"]} <= set(expected.order_id)
    assert answer["table"]["truncated"] == (len(expected) > 5)
    assert answer["chart"] is None
    assert "Rows showing order_id, customer, revenue" in answer["explanation"]["description"]


def test_what_is_this_file_about_is_answered_from_the_profile(
    app: FastAPI, client: TestClient, acct: Account, dataset_id: str
) -> None:
    use_llm(app, FakeLLM([{"metric": "describe"}], summary="Sales orders by customer."))
    reply = ask(client, acct, "what is this file about?", dataset_id=dataset_id).json()["reply"]
    table = reply["answer"]["table"]
    assert reply["status"] == "answered"
    assert table["columns"] == ["column", "type", "examples or range"]
    assert ["revenue", "number", "16.15 to 12,475"] in table["rows"]
    assert reply["answer"]["explanation"]["sql"].startswith("-- no query")


def test_rows_cannot_be_grouped(
    app: FastAPI, client: TestClient, acct: Account, dataset_id: str
) -> None:
    bad = {"metric": "rows", "group_by": ["region"]}
    use_llm(app, FakeLLM([bad, bad]))
    reply = ask(client, acct, "list by region", dataset_id=dataset_id).json()["reply"]
    assert reply["status"] == "failed"


# --- streaming progress ------------------------------------------------------------------


def stream_events(client: TestClient, acct: Account, **body: str) -> list[dict[str, Any]]:
    import json as _json

    with client.stream("POST", "/chat/ask/stream", headers=acct.headers, json=body) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        lines = [ln for ln in resp.iter_lines() if ln.startswith("data: ")]
    return [_json.loads(ln.removeprefix("data: ")) for ln in lines]


def test_stream_sends_stages_then_the_answer(
    app: FastAPI, client: TestClient, acct: Account, dataset_id: str
) -> None:
    use_llm(app, FakeLLM([REVENUE_BY_REGION], summary=first_row_summary))
    events = stream_events(client, acct, question="Revenue by region", dataset_id=dataset_id)
    assert [e["stage"] for e in events if e["type"] == "stage"] == [
        "planning", "querying", "summarizing"
    ]  # fmt: skip
    done = events[-1]
    assert done["type"] == "done"
    assert done["response"]["reply"]["answer"]["table"]["columns"] == ["region", "sum_revenue"]
    # stored like a normal answer
    conv_id = done["response"]["conversation_id"]
    conv = client.get(f"/chat/conversations/{conv_id}", headers=acct.headers).json()
    assert len(conv["messages"]) == 2


def test_stream_reports_errors_as_events(
    app: FastAPI, client: TestClient, acct: Account, dataset_id: str
) -> None:
    use_llm(app, FakeLLM(unavailable=True))
    events = stream_events(client, acct, question="anything", dataset_id=dataset_id)
    assert events[-1] == {
        "type": "error",
        "status": 503,
        "detail": "The fake model is switched off.",
    }
    other = stream_events(client, acct, question="x", dataset_id="nope")
    assert other[-1]["status"] == 404


# --- "write python code" and "download" ----------------------------------------------


def test_python_code_reproduces_the_last_answer(
    app: FastAPI, client: TestClient, acct: Account, dataset_id: str
) -> None:
    llm = use_llm(app, FakeLLM([REVENUE_BY_REGION], summary="Done."))
    conv = ask(client, acct, "Revenue by region", dataset_id=dataset_id).json()["conversation_id"]
    calls = len(llm.calls)
    reply = ask(client, acct, "Can you write python code?", conversation_id=conv).json()["reply"]
    assert len(llm.calls) == calls  # generated from the plan, no model call
    assert reply["answer"]["type"] == "code"
    python = reply["answer"]["blocks"][0]
    assert python["language"] == "python"
    assert reply["answer"]["blocks"][1]["language"] == "sql"

    import os

    namespace: dict[str, Any] = {}
    cwd = os.getcwd()
    os.chdir(SAMPLE)
    try:
        exec(python["code"].replace("sales_orders.csv", "sales_orders.csv"), namespace)  # noqa: S102
    finally:
        os.chdir(cwd)
    expected = pd.read_csv(SAMPLE / "sales_orders.csv").groupby("region")["revenue"].sum()
    got = namespace["result"].set_index("region")["sum_revenue"]
    assert got.to_dict() == pytest.approx(expected.to_dict())


def test_code_before_any_question_explains(
    app: FastAPI, client: TestClient, acct: Account, dataset_id: str
) -> None:
    use_llm(app, FakeLLM([]))
    reply = ask(client, acct, "write python code", dataset_id=dataset_id).json()["reply"]
    assert "Ask me a question about the data first" in reply["content"]


def test_download_request_points_to_buttons(
    app: FastAPI, client: TestClient, acct: Account, dataset_id: str
) -> None:
    llm = use_llm(app, FakeLLM([]))
    reply = ask(client, acct, "give me a download option", dataset_id=dataset_id).json()["reply"]
    assert "CSV" in reply["content"]
    assert llm.calls == []


# --- styled Excel -------------------------------------------------------------------------


def test_excel_request_gives_a_styled_workbook(
    app: FastAPI, client: TestClient, acct: Account, dataset_id: str
) -> None:
    import io

    from openpyxl import load_workbook

    llm = use_llm(app, FakeLLM([REVENUE_BY_REGION], summary="North leads."))
    conv = ask(client, acct, "Revenue by region", dataset_id=dataset_id).json()["conversation_id"]
    calls = len(llm.calls)
    reply = ask(client, acct, "ok good but give me formatted and styled excel",
                conversation_id=conv).json()["reply"]  # fmt: skip
    assert len(llm.calls) == calls  # no model call
    assert reply["answer"]["type"] == "file"
    answer_id = reply["answer"]["message_id"]

    resp = client.get(f"/chat/messages/{answer_id}/export.xlsx", headers=acct.headers)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/vnd.openxmlformats")
    assert 'filename="Revenue-by-region.xlsx"' in resp.headers["content-disposition"]

    ws = load_workbook(io.BytesIO(resp.content)).active
    assert ws is not None
    assert ws["A1"].value == "Revenue by region"
    assert ws["A2"].value == "North leads."
    assert [c.value for c in ws[5]][:2] == ["region", "Total revenue"]
    assert ws["A5"].font.bold
    assert ws["A5"].fill.fgColor.rgb.endswith("4F46E5")
    assert ws["B6"].number_format == "#,##0.00"
    assert isinstance(ws["B6"].value, float)
    assert ws.auto_filter.ref == "A5:B9"
    assert ws.freeze_panes == "A6"
    assert len(ws._charts) == 1  # type: ignore[attr-defined]  # native Excel chart

    other = signup(client, "spy@other.com", workspace="Other")
    assert (
        client.get(f"/chat/messages/{answer_id}/export.xlsx", headers=other.headers).status_code
        == 404
    )


def test_excel_cells_never_become_formulas() -> None:
    import io
    from datetime import UTC, datetime

    from openpyxl import load_workbook

    from app.exporting import build_answer_workbook

    data = build_answer_workbook(
        question="q", summary="s", chart=None, dataset_name="d", created_at=datetime.now(UTC),
        table={"columns": ["name", "when"], "rows": [["=HYPERLINK(\"evil\")", "2026-01-02"]]},
    )  # fmt: skip
    ws = load_workbook(io.BytesIO(data)).active
    assert ws is not None
    assert ws["A6"].value == '\'=HYPERLINK("evil")'
    assert ws["B6"].number_format == "yyyy-mm-dd"


def test_excel_before_any_answer_explains(
    app: FastAPI, client: TestClient, acct: Account, dataset_id: str
) -> None:
    use_llm(app, FakeLLM([]))
    reply = ask(client, acct, "give me an excel file", dataset_id=dataset_id).json()["reply"]
    assert "Ask me for the data first" in reply["content"]
