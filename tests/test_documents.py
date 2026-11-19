"""Document mode: passages, search, routing, and cited answers that are checked."""

import io
import json
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fpdf import FPDF

from app.documents.answer import NOT_FOUND, UNGROUNDED
from app.documents.ingest import Passage, chunk, ingest_document, load_passages
from app.documents.search import search
from app.llm.factory import get_llm
from app.query.presentation import text_is_grounded
from tests.conftest import Account, signup
from tests.fakes import FakeLLM

POLICY = [
    "Leave policy. Every permanent employee is entitled to 24 days of annual leave per year.",
    "Remote work. Staff may work from home up to 2 days per week with manager approval.",
    "Expenses. Travel claims must be submitted within 30 days, with receipts above PKR 5,000.",
    "Probation lasts 3 months for all new hires, after which a review is held.",
]


def docx_bytes(paragraphs: list[str]) -> bytes:
    import docx

    document = docx.Document()
    for p in paragraphs:
        document.add_paragraph(p)
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text, table.cell(0, 1).text = "Grade", "Allowance"
    table.cell(1, 0).text, table.cell(1, 1).text = "G5", "12,000"
    buf = io.BytesIO()
    document.save(buf)
    return buf.getvalue()


def prose_pdf(pages: list[str]) -> bytes:
    pdf = FPDF()
    pdf.set_font("Helvetica", size=11)
    for text in pages:
        pdf.add_page()
        pdf.multi_cell(0, 6, text)
    return bytes(pdf.output())


# --- passages and search -------------------------------------------------------------------


def test_chunks_keep_pages_and_stay_bounded() -> None:
    long_page = " ".join(f"Sentence number {i} about leave rules." for i in range(200))
    passages = chunk([(1, "Short first page."), (2, long_page)])
    assert passages[0].page == 1
    assert all(p.page == 2 for p in passages[1:])
    assert all(len(p.text) <= 1000 for p in passages)
    assert [p.chunk_id for p in passages] == list(range(1, len(passages) + 1))
    # overlap: consecutive passages share some words, so answers aren't cut in half
    assert passages[2].text.split()[0] in passages[1].text


def test_search_ranks_the_relevant_passage_first() -> None:
    passages = [Passage(i, None, t) for i, t in enumerate(POLICY, start=1)]
    assert search("How many days of annual leave do employees get?", passages)[0].chunk_id == 1
    assert search("work from home", passages)[0].chunk_id == 2
    assert search("quarterly bonus formula xyz", passages) == []


@pytest.mark.parametrize(
    ("answer", "ok"),
    [
        ("Employees get 24 days of annual leave [1].", True),
        ("Receipts are needed above PKR 5,000 [3].", True),
        ("Employees get 30 days of annual leave [1].", True),  # 30 is in the sources (claims)
        ("Employees get 26 days of annual leave [1].", False),  # invented
        ("Probation lasts 6 months.", False),
    ],
)
def test_text_grounding(answer: str, ok: bool) -> None:
    assert text_is_grounded(answer, POLICY, "leave?") is ok


def test_ingest_and_reload_docx(tmp_path: Path) -> None:
    src = tmp_path / "policy.docx"
    src.write_bytes(docx_bytes(POLICY))
    result = ingest_document(src, "docx", tmp_path / "out.parquet")
    assert result.passages >= 1
    text = " ".join(p.text for p in load_passages(tmp_path / "out.parquet"))
    assert "24 days" in text
    assert "G5 | 12,000" in text  # Word tables stay searchable


# --- upload routing ----------------------------------------------------------------------


@pytest.fixture
def acct(client: TestClient) -> Account:
    return signup(client, "docs@acme.com")


def upload(client: TestClient, acct: Account, name: str, content: bytes, **form: str) -> Any:
    return client.post(
        "/datasets", headers=acct.headers, files={"file": (name, content)}, data=form
    )


def table_pdf() -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)
    with pdf.table() as table:
        for row in [["region", "revenue"], ["North", "10"], ["South", "20"], ["East", "5"]]:
            r = table.row()
            for cell in row:
                r.cell(cell)
    return bytes(pdf.output())


@pytest.mark.parametrize(
    ("name", "content", "form", "kind"),
    [
        ("policy.docx", docx_bytes(POLICY), {}, "document"),
        ("notes.md", "# Policy\n\n" + "\n\n".join(POLICY).encode().decode(), {}, "document"),
        ("letter.txt", "\n\n".join(POLICY), {}, "document"),  # prose
        ("data.txt", "region\tsales\nNorth\t10\nSouth\t20\n", {}, "table"),  # delimited
        ("handbook.pdf", prose_pdf(POLICY), {}, "document"),
        ("report.pdf", table_pdf(), {}, "table"),
        ("report.pdf", table_pdf(), {"mode": "document"}, "document"),  # forced
    ],
)
def test_upload_routing(
    client: TestClient, acct: Account, name: str, content: Any, form: dict[str, str], kind: str
) -> None:
    data = content.encode() if isinstance(content, str) else content
    resp = upload(client, acct, name, data, **form)
    assert resp.status_code == 201, resp.text
    assert resp.json()["kind"] == kind
    if kind == "document":
        assert resp.json()["columns"] == []
        assert resp.json()["row_count"] >= 1  # passages


@pytest.mark.parametrize(
    ("name", "form", "message"),
    [
        ("policy.docx", {"mode": "table"}, "read as documents"),
        ("sales.csv", {"mode": "document"}, "can be read as documents"),
    ],
)
def test_upload_mode_conflicts(
    client: TestClient, acct: Account, name: str, form: dict[str, str], message: str
) -> None:
    content = docx_bytes(POLICY) if name.endswith(".docx") else b"a,b\n1,2\n"
    resp = upload(client, acct, name, content, **form)
    assert resp.status_code == 400
    assert message in resp.json()["detail"]


# --- asking ------------------------------------------------------------------------------


@pytest.fixture
def doc_id(client: TestClient, acct: Account) -> str:
    resp = upload(client, acct, "policy.docx", docx_bytes(POLICY))
    assert resp.status_code == 201, resp.text
    return str(resp.json()["id"])


def ask(client: TestClient, acct: Account, question: str, **ids: str) -> Any:
    resp = client.post("/chat/ask", headers=acct.headers, json={"question": question, **ids})
    assert resp.status_code == 200, resp.text
    return resp.json()


def reply_for(model_reply: dict[str, Any]) -> FakeLLM:
    return FakeLLM([json.dumps(model_reply)])


def test_answer_with_cited_sources(
    app: FastAPI, client: TestClient, acct: Account, doc_id: str
) -> None:
    llm = reply_for({"found": True, "answer": "Employees get 24 days a year [1].", "sources": [1]})
    app.dependency_overrides[get_llm] = lambda: llm
    body = ask(client, acct, "How much annual leave do employees get?", dataset_id=doc_id)
    reply = body["reply"]
    assert reply["status"] == "answered"
    assert reply["content"] == "Employees get 24 days a year [1]."
    answer = reply["answer"]
    assert answer["type"] == "document"
    assert answer["grounded"] is True
    assert "24 days" in answer["sources"][0]["text"]
    assert answer["sources"][0]["cited"] is True
    # the model saw numbered passages, never the whole document
    prompt = llm.plan_calls[0][-1].content
    assert prompt.startswith("Passages:\n[1]")
    # stored and re-read through the conversation API
    conv = client.get(f"/chat/conversations/{body['conversation_id']}", headers=acct.headers)
    assert conv.json()["messages"][1]["answer"]["type"] == "document"


def test_invented_number_is_withheld(
    app: FastAPI, client: TestClient, acct: Account, doc_id: str
) -> None:
    app.dependency_overrides[get_llm] = lambda: reply_for(
        {"found": True, "answer": "Employees get 36 days of leave [1].", "sources": [1]}
    )
    reply = ask(client, acct, "How much annual leave?", dataset_id=doc_id)["reply"]
    assert reply["content"] == UNGROUNDED
    assert reply["answer"]["grounded"] is False
    assert reply["answer"]["sources"]  # the relevant passages are still shown


def test_not_in_document(app: FastAPI, client: TestClient, acct: Account, doc_id: str) -> None:
    # No shared keywords: the model still reads the document, and says it isn't there.
    app.dependency_overrides[get_llm] = lambda: reply_for(
        {"found": False, "answer": "", "sources": []}
    )
    reply = ask(client, acct, "quarterly bonus formula xyz", dataset_id=doc_id)["reply"]
    assert reply["status"] == "failed"
    assert reply["content"] == NOT_FOUND

    app.dependency_overrides[get_llm] = lambda: reply_for(
        {"found": False, "answer": "", "sources": []}
    )
    reply = ask(client, acct, "What is the leave carry-over rule?", dataset_id=doc_id)["reply"]
    assert reply["content"] == NOT_FOUND


def test_follow_up_gets_previous_turn(
    app: FastAPI, client: TestClient, acct: Account, doc_id: str
) -> None:
    llm = FakeLLM(
        [
            json.dumps({"found": True, "answer": "24 days [1].", "sources": [1]}),
            json.dumps({"found": True, "answer": "Up to 2 days per week [1].", "sources": [1]}),
        ]
    )
    app.dependency_overrides[get_llm] = lambda: llm
    first = ask(client, acct, "How much annual leave?", dataset_id=doc_id)
    ask(client, acct, "and remote work from home?", conversation_id=first["conversation_id"])
    follow_up = llm.plan_calls[1]
    assert any(m.content == "Question: How much annual leave?" for m in follow_up)
    assert any(m.role == "assistant" and m.content == "24 days [1]." for m in follow_up)


def test_what_is_this_document_about_reads_the_whole_document(
    app: FastAPI, client: TestClient, acct: Account, doc_id: str
) -> None:
    llm = reply_for({"found": True, "answer": "It is the company handbook [1].", "sources": [1]})
    app.dependency_overrides[get_llm] = lambda: llm
    reply = ask(client, acct, "What is this document about?", dataset_id=doc_id)["reply"]
    assert reply["status"] == "answered"
    assert reply["content"] == "It is the company handbook [1]."
    assert len(llm.calls) == 1  # answered from passages across the document


def test_search_matches_word_forms() -> None:
    from app.documents.search import tokenize

    assert set(tokenize("types of normalizations")) & set(tokenize("Normal forms: 1NF, 2NF"))


def test_explain_more_follow_up_expands_the_previous_answer(
    app: FastAPI, client: TestClient, acct: Account, doc_id: str
) -> None:
    llm = FakeLLM([
        json.dumps({"found": True, "answer": "It is the company handbook [1].", "sources": [1]}),
        json.dumps({"found": True, "answer": "It covers leave and benefits [1].", "sources": [1]}),
    ])  # fmt: skip
    app.dependency_overrides[get_llm] = lambda: llm
    first = client.post(
        "/chat/ask",
        headers=acct.headers,
        json={"question": "What is this document about?", "dataset_id": doc_id},
    )
    conv = first.json()["conversation_id"]
    reply = ask(client, acct, "can you explain more?", conversation_id=conv)["reply"]
    assert reply["status"] == "answered"
    assert reply["content"] == "It covers leave and benefits [1]."
    # The model saw the earlier turn, so "more" means more about the same thing.
    sent = " ".join(m.content for m in llm.calls[-1][0])
    assert "What is this document about?" in sent
