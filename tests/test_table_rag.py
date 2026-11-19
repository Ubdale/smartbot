"""Table RAG: real values for loose names, topic filters by meaning, and proof rows."""

import json
import re
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from app.api.deps import get_today
from app.core.config import Settings
from app.llm.base import ChatMessage, Completion
from app.llm.factory import get_llm
from app.query.executor import clear_cache
from app.query.retrieval import closest_values, match_topic, match_values
from tests.conftest import Account, signup
from tests.fakes import FakeLLM

TODAY = date(2026, 11, 19)
SAMPLE = Path(__file__).resolve().parent.parent / "sample_data"
TASKS_CSV = """Issue key,Assignee,Status,Summary,Created
DMA-1,Zarmeen Khan,Open,Fix Tax Liability VS Tax Paid,2026-10-01
DMA-2,Sara Ahmed,Done,VAT return mismatch in October,2026-10-03
DMA-3,Sara Khan,Open,Redirect to Analytics (Beta 501),2026-10-05
DMA-4,Zarmeen Khan,Open,Withholding tax report shows wrong totals,2026-10-07
DMA-5,Muhammad Ubdale,Done,EDM UAT Points,2026-10-09
DMA-6,Sara Ahmed,Open,Make API for setting up online status,2026-10-11
DMA-7,Areesha Idrees,Open,Payroll export misses overtime,2026-10-13
"""
TAXY = re.compile(r"tax|vat|withholding", re.I)


class MeaningLLM(FakeLLM):
    """Plans from a script; for "which texts are about X" it picks tax-related lines."""

    def complete(self, messages: list[ChatMessage], *, json_schema: dict[str, Any] | None = None,
                 max_tokens: int = 512) -> Completion:  # fmt: skip
        last = messages[-1].content
        if json_schema is not None and last.startswith("Topic:") and "Texts:" in last:
            self.calls.append((messages, json_schema))
            picked = [int(n) for n, text in re.findall(r"^(\d+)\. (.*)$", last, re.M)
                      if TAXY.search(text)]  # fmt: skip
            return Completion(json.dumps({"matches": picked}), 50, 10, 5)
        return super().complete(messages, json_schema=json_schema, max_tokens=max_tokens)


@pytest.fixture(autouse=True)
def _setup(app: FastAPI) -> Iterator[None]:
    clear_cache()
    app.dependency_overrides[get_today] = lambda: TODAY
    yield
    clear_cache()


@pytest.fixture
def acct(client: TestClient) -> Account:
    return signup(client, "owner@acme.com", workspace="Acme")


def upload(client: TestClient, acct: Account, name: str, content: bytes) -> str:
    resp = client.post("/datasets", headers=acct.headers, files={"file": (name, content)})
    assert resp.status_code == 201, resp.text
    return str(resp.json()["id"])


def ask(client: TestClient, acct: Account, question: str, **ids: str) -> dict[str, Any]:
    resp = client.post("/chat/ask", headers=acct.headers, json={"question": question, **ids})
    assert resp.status_code == 200, resp.text
    reply: dict[str, Any] = resp.json()["reply"]
    return reply


def test_closest_values() -> None:
    names = ["Zarmeen Khan", "Sara Ahmed", "Sara Khan", "Muhammad Ubdale"]
    assert closest_values("zarmeen", names) == ["Zarmeen Khan"]
    assert closest_values("SARA AHMED", names) == ["Sara Ahmed"]
    assert sorted(closest_values("Sara", names)) == ["Sara Ahmed", "Sara Khan"]
    assert closest_values("Bilal", names) == []


def _parquet(
    settings: Settings, session_factory: sessionmaker[Session], ds: str
) -> tuple[Path, list[dict[str, Any]]]:
    from app.api.datasets import require_parquet
    from app.models import Dataset

    with session_factory() as db:
        dataset = db.get(Dataset, ds)
        assert dataset is not None
        return require_parquet(settings, dataset), dataset.columns


def test_value_hints_find_real_names(
    client: TestClient, acct: Account, settings: Settings,
    session_factory: sessionmaker[Session],
) -> None:  # fmt: skip
    ds = upload(client, acct, "tasks.csv", TASKS_CSV.encode())
    parquet, profile = _parquet(settings, session_factory, ds)
    found = match_values("open tasks of zarmeen and areesha", parquet, profile)
    assert {(m.column, m.value) for m in found} >= {("Assignee", "Zarmeen Khan"),
                                                    ("Assignee", "Areesha Idrees")}  # fmt: skip


def test_loose_names_are_matched_to_real_values(
    app: FastAPI, client: TestClient, acct: Account
) -> None:
    ds = upload(client, acct, "tasks.csv", TASKS_CSV.encode())
    plan = {"metric": "count", "filters": [{"column": "Assignee", "op": "=", "value": "zarmeen"}]}
    sara = {"metric": "count", "filters": [{"column": "Assignee", "op": "=", "value": "Sara"}]}
    llm = FakeLLM([plan, sara], summary="Done.")
    app.dependency_overrides[get_llm] = lambda: llm

    reply = ask(client, acct, "how many tasks does zarmeen have", dataset_id=ds)
    assert reply["answer"]["table"]["rows"] == [[2]]
    notes = reply["answer"]["explanation"]["retrieval_notes"]
    assert notes == ["Matched “zarmeen” to “Zarmeen Khan” in Assignee."]
    # The planner was shown the real name before planning.
    assert "Zarmeen Khan" in llm.plan_calls[0][-1].content or any(
        "Zarmeen Khan" in m.content for m in llm.plan_calls[0]
    )

    # A name that fits two people counts both, and says so.
    reply = ask(client, acct, "how many tasks does Sara have", dataset_id=ds)
    assert reply["answer"]["table"]["rows"] == [[3]]
    assert "Sara Ahmed" in reply["answer"]["explanation"]["retrieval_notes"][0]


def test_topic_filter_counts_rows_by_meaning_and_shows_proof(
    app: FastAPI, client: TestClient, acct: Account
) -> None:
    ds = upload(client, acct, "tasks.csv", TASKS_CSV.encode())
    plan = {"metric": "count", "group_by": ["Assignee"], "chart": "bar",
            "filters": [{"column": "Summary", "op": "about", "value": "tax issues"}]}  # fmt: skip
    llm = MeaningLLM([plan], summary="Zarmeen Khan has the most tax tasks.")
    app.dependency_overrides[get_llm] = lambda: llm
    reply = ask(client, acct, "how many tasks are about tax issues, per person?", dataset_id=ds)

    rows = {r[0]: r[1] for r in reply["answer"]["table"]["rows"]}
    assert rows == {"Zarmeen Khan": 2, "Sara Ahmed": 1}  # "VAT return" counts: meaning
    explanation = reply["answer"]["explanation"]
    assert "the AI read all 7 Summary texts; 3 rows match" in explanation["retrieval_notes"][0]
    assert sorted(explanation["topic_rows"]["0"]) == [0, 1, 3]
    assert "Summary is about “tax issues”" in explanation["description"]

    # Proof: every row behind the answer, then just one person's rows.
    resp = client.get(f"/chat/messages/{reply['id']}/evidence", headers=acct.headers)
    assert resp.status_code == 200, resp.text
    ev = resp.json()
    assert ev["total"] == 3
    keys = ev["columns"].index("Issue key")
    assert sorted(r[keys] for r in ev["rows"]) == ["DMA-1", "DMA-2", "DMA-4"]
    assert "file_row_number" not in ev["columns"]

    one = {"group": json.dumps(["Zarmeen Khan"])}
    resp = client.get(f"/chat/messages/{reply['id']}/evidence", params=one, headers=acct.headers)
    assert resp.json()["total"] == 2
    other = signup(client, "spy@other.com", workspace="Other")
    assert client.get(f"/chat/messages/{reply['id']}/evidence",
                      headers=other.headers).status_code == 404  # fmt: skip


def test_topic_falls_back_to_keywords_without_the_model(
    client: TestClient, acct: Account, settings: Settings,
    session_factory: sessionmaker[Session],
) -> None:  # fmt: skip
    ds = upload(client, acct, "tasks.csv", TASKS_CSV.encode())
    parquet, _profile = _parquet(settings, session_factory, ds)
    m = match_topic(FakeLLM(unavailable=True), parquet, "Summary", "tax")
    assert m.method == "keywords"
    assert sorted(m.rows) == [0, 3]  # literal "tax" only; "VAT" needs the model


def test_evidence_for_one_month_of_a_trend(app: FastAPI, client: TestClient, acct: Account) -> None:
    ds = upload(client, acct, "sales_orders.csv", (SAMPLE / "sales_orders.csv").read_bytes())
    plan = {"metric": "count", "time_grain": "month", "date_column": "order_date",
            "date_phrase": "2024", "limit": 100, "chart": "line"}  # fmt: skip
    app.dependency_overrides[get_llm] = lambda: FakeLLM([plan], summary="Steady.")
    reply = ask(client, acct, "orders per month in 2024", dataset_id=ds)
    month, count = reply["answer"]["table"]["rows"][2]
    resp = client.get(f"/chat/messages/{reply['id']}/evidence",
                      params={"group": json.dumps([month])}, headers=acct.headers)  # fmt: skip
    ev = resp.json()
    assert ev["total"] == count
    dates = {r[ev["columns"].index("order_date")][:7] for r in ev["rows"]}
    assert dates == {month[:7]}
