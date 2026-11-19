"""AskData learns from use: 👍 examples, team vocabulary, and instant repeats."""

from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.deps import get_today
from app.learning import parse_definition
from app.llm.factory import get_llm
from app.query.executor import clear_cache
from tests.conftest import Account, signup
from tests.fakes import FakeLLM

SAMPLE = Path(__file__).resolve().parent.parent / "sample_data"
BY_REGION = {"metric": "sum", "metric_column": "revenue", "group_by": ["region"]}


@pytest.fixture(autouse=True)
def _setup(app: FastAPI) -> Iterator[None]:
    clear_cache()
    app.dependency_overrides[get_today] = lambda: date(2026, 11, 19)
    yield
    clear_cache()


@pytest.fixture
def acct(client: TestClient) -> Account:
    return signup(client, "learner@acme.com")


@pytest.fixture
def ds(client: TestClient, acct: Account) -> str:
    content = (SAMPLE / "sales_orders.csv").read_bytes()
    resp = client.post("/datasets", headers=acct.headers, files={"file": ("sales.csv", content)})
    return str(resp.json()["id"])


def ask(client: TestClient, acct: Account, q: str, **ids: str) -> dict[str, Any]:
    resp = client.post("/chat/ask", headers=acct.headers, json={"question": q, **ids})
    assert resp.status_code == 200, resp.text
    return dict(resp.json())


def rate(client: TestClient, acct: Account, message_id: str, rating: int) -> dict[str, Any]:
    resp = client.post(
        f"/chat/messages/{message_id}/feedback", headers=acct.headers, json={"rating": rating}
    )
    assert resp.status_code == 200, resp.text
    return dict(resp.json())


# --- vocabulary -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("sales means revenue", ("sales", "revenue")),
        ("By staff I mean headcount", ("staff", "headcount")),
        ("GP = gross profit", ("GP", "gross profit")),
        ('"AOV" stands for average order value.', ("AOV", "average order value")),
        ("what does sales mean?", None),  # a question, not a definition
        ("show me revenue by region", None),
        ("this column means nothing to me but I would like a long explanation of it all", None),
    ],
)
def test_parse_definition(text: str, expected: tuple[str, str] | None) -> None:
    d = parse_definition(text)
    assert (d.term, d.meaning) == expected if d else expected is None


def test_teaching_a_word_needs_no_model_and_is_used_later(
    app: FastAPI, client: TestClient, acct: Account, ds: str
) -> None:
    llm = FakeLLM([BY_REGION], summary="Done.")
    app.dependency_overrides[get_llm] = lambda: llm
    taught = ask(client, acct, "sales means revenue", dataset_id=ds)["reply"]
    assert taught["content"].startswith("Got it.")
    assert llm.calls == []  # remembered without asking the model

    glossary = client.get(f"/datasets/{ds}/glossary", headers=acct.headers).json()
    assert [(g["term"], g["meaning"]) for g in glossary] == [("sales", "revenue")]

    ask(client, acct, "sales by region", dataset_id=ds)
    prompt = llm.plan_calls[0][-1].content
    assert 'Words this team uses: "sales" means revenue.' in prompt


def test_glossary_endpoints_and_isolation(client: TestClient, acct: Account, ds: str) -> None:
    added = client.post(
        f"/datasets/{ds}/glossary", headers=acct.headers, json={"term": "AOV", "meaning": "avg"}
    ).json()
    assert [g["term"] for g in added] == ["AOV"]
    other = signup(client, "spy@other.com", workspace="Other")
    assert client.get(f"/datasets/{ds}/glossary", headers=other.headers).status_code == 404
    term_id = added[0]["id"]
    assert (
        client.delete(f"/datasets/{ds}/glossary/{term_id}", headers=other.headers).status_code
        == 404
    )
    assert (
        client.delete(f"/datasets/{ds}/glossary/{term_id}", headers=acct.headers).status_code == 204
    )
    assert client.get(f"/datasets/{ds}/glossary", headers=acct.headers).json() == []


# --- 👍 examples ------------------------------------------------------------------------


def test_thumbs_up_teaches_similar_questions(
    app: FastAPI, client: TestClient, acct: Account, ds: str
) -> None:
    llm = FakeLLM([BY_REGION, BY_REGION], summary="Done.")
    app.dependency_overrides[get_llm] = lambda: llm
    first = ask(client, acct, "total revenue per region", dataset_id=ds)
    result = rate(client, acct, first["reply"]["id"], 1)
    assert result == {"message_id": first["reply"]["id"], "rating": 1, "learned": True}

    ask(client, acct, "revenue per region this year", dataset_id=ds)  # similar, not identical
    history = [m.content for m in llm.plan_calls[1][1:-1]]
    assert "Question: total revenue per region" in history
    assert any('"group_by":["region"]' in h for h in history)


def test_thumbs_down_unlearns(app: FastAPI, client: TestClient, acct: Account, ds: str) -> None:
    llm = FakeLLM([BY_REGION, BY_REGION], summary="Done.")
    app.dependency_overrides[get_llm] = lambda: llm
    first = ask(client, acct, "total revenue per region", dataset_id=ds)
    rate(client, acct, first["reply"]["id"], 1)
    assert rate(client, acct, first["reply"]["id"], -1)["learned"] is False
    ask(client, acct, "revenue per region this year", dataset_id=ds)
    assert all("total revenue per region" not in m.content for m in llm.plan_calls[1])


def test_cannot_rate_someone_elses_answer(
    app: FastAPI, client: TestClient, acct: Account, ds: str
) -> None:
    app.dependency_overrides[get_llm] = lambda: FakeLLM([BY_REGION], summary="Done.")
    reply_id = ask(client, acct, "revenue by region", dataset_id=ds)["reply"]["id"]
    other = signup(client, "other@x.com", workspace="X")
    resp = client.post(
        f"/chat/messages/{reply_id}/feedback", headers=other.headers, json={"rating": 1}
    )
    assert resp.status_code == 404


# --- instant repeats --------------------------------------------------------------------


def test_identical_question_is_answered_without_the_model(
    app: FastAPI, client: TestClient, acct: Account, ds: str
) -> None:
    llm = FakeLLM([BY_REGION], summary="North leads.")
    app.dependency_overrides[get_llm] = lambda: llm
    first = ask(client, acct, "Revenue by region", dataset_id=ds)["reply"]
    calls_after_first = len(llm.calls)

    again = ask(client, acct, "  revenue BY region ", dataset_id=ds)["reply"]
    assert len(llm.calls) == calls_after_first  # no model call
    assert again["answer"]["reused"] is True
    assert again["answer"]["table"] == first["answer"]["table"]  # recomputed, same numbers
    assert again["content"] == "North leads."


def test_wrong_answers_are_never_reused(
    app: FastAPI, client: TestClient, acct: Account, ds: str
) -> None:
    llm = FakeLLM([BY_REGION, {"metric": "count"}], summary="Done.")
    app.dependency_overrides[get_llm] = lambda: llm
    first = ask(client, acct, "Revenue by region", dataset_id=ds)["reply"]
    rate(client, acct, first["id"], -1)
    again = ask(client, acct, "Revenue by region", dataset_id=ds)["reply"]
    assert again["answer"]["reused"] is False
    assert len(llm.plan_calls) == 2  # asked the model again


def test_follow_ups_are_never_reused(
    app: FastAPI, client: TestClient, acct: Account, ds: str
) -> None:
    llm = FakeLLM([BY_REGION, {"metric": "count"}, BY_REGION], summary="Done.")
    app.dependency_overrides[get_llm] = lambda: llm
    conv = ask(client, acct, "Revenue by region", dataset_id=ds)["conversation_id"]
    ask(client, acct, "how many orders?", conversation_id=conv)
    follow = ask(client, acct, "Revenue by region", conversation_id=conv)["reply"]
    assert follow["answer"]["reused"] is False  # context-dependent: asked the model
