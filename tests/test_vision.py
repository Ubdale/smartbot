"""Tables read from images (and scanned PDFs) by a vision model, marked "please check"."""

import base64
import io
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fpdf import FPDF
from PIL import Image
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.api.datasets import get_vision_llm
from app.core.config import Settings
from app.llm.base import ChatMessage, Completion, LLMProvider, LLMUnavailableError
from app.llm.factory import get_llm, make_vision_provider
from app.llm.gemini_provider import GeminiProvider
from app.llm.ollama_provider import OllamaProvider
from app.models import UsageEvent
from tests.conftest import Account, signup
from tests.fakes import FakeLLM

TABLE = [["Branch", "Staff"], ["Karachi", "2,579"], ["Lahore", "851"], ["Total", "3,430"]]


class FakeVision(LLMProvider):
    name, model = "fake-vision", "eyes-1"

    def __init__(self, rows: list[list[str]] | None = None, reply: str | None = None) -> None:
        self.rows = rows if rows is not None else TABLE
        self.reply = reply
        self.images: list[tuple[int, str]] = []

    def complete(self, messages: list[ChatMessage], **kw: Any) -> Completion:
        raise AssertionError("text completion not expected")

    def complete_with_image(self, prompt: str, image: bytes, mime: str, **kw: Any) -> Completion:
        self.images.append((len(image), mime))
        text = self.reply if self.reply is not None else json.dumps({"rows": self.rows})
        return Completion(text, 1200, 80, 900)


def png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (20, 10), "white").save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def acct(client: TestClient) -> Account:
    return signup(client, "eyes@acme.com")


def upload(client: TestClient, acct: Account, name: str, content: bytes) -> Any:
    return client.post("/datasets", headers=acct.headers, files={"file": (name, content)})


# --- upload flow ------------------------------------------------------------------------------


def test_image_table_becomes_a_checked_dataset(
    app: FastAPI, client: TestClient, acct: Account, session_factory: sessionmaker[Session]
) -> None:
    vision = FakeVision()
    app.dependency_overrides[get_vision_llm] = lambda: vision
    resp = upload(client, acct, "branches.png", png_bytes())
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["extraction"] == "ai_vision"
    assert body["row_count"] == 2  # Total row dropped, like any report
    cols = {c["name"]: c for c in body["columns"]}
    assert cols["Staff"]["kind"] == "number"
    assert cols["Staff"]["max"] == 2579  # "2,579" typed by code, not by the model
    assert vision.images[0][1] == "image/png"
    with session_factory() as db:
        events = list(db.scalars(select(UsageEvent)))
    assert [(e.purpose, e.provider) for e in events] == [("vision", "fake-vision")]


def test_answers_on_image_data_carry_a_note(
    app: FastAPI, client: TestClient, acct: Account
) -> None:
    app.dependency_overrides[get_vision_llm] = lambda: FakeVision()
    ds = upload(client, acct, "branches.jpg", png_bytes()).json()
    plan = {"metric": "sum", "metric_column": "Staff", "group_by": ["Branch"]}
    app.dependency_overrides[get_llm] = lambda: FakeLLM([plan], summary="Done.")
    reply = client.post(
        "/chat/ask",
        headers=acct.headers,
        json={"question": "staff by branch", "dataset_id": ds["id"]},
    ).json()["reply"]
    assert reply["answer"]["table"]["rows"] == [["Karachi", 2579], ["Lahore", 851]]
    assert "read from an image by AI" in reply["answer"]["explanation"]["data_note"]


def test_image_without_vision_model_is_explained(
    app: FastAPI, client: TestClient, acct: Account
) -> None:
    app.dependency_overrides[get_vision_llm] = lambda: None
    resp = upload(client, acct, "x.png", png_bytes())
    assert resp.status_code == 400
    assert "vision model" in resp.json()["detail"]


@pytest.mark.parametrize(
    ("vision", "message"),
    [
        (FakeVision(rows=[]), "No table found"),
        (FakeVision(reply="sorry, I can't"), "couldn't read a table"),
    ],
)
def test_unreadable_images(
    app: FastAPI, client: TestClient, acct: Account, vision: FakeVision, message: str
) -> None:
    app.dependency_overrides[get_vision_llm] = lambda: vision
    resp = upload(client, acct, "x.webp", png_bytes())
    assert resp.status_code == 400
    assert message in resp.json()["detail"]


def test_scanned_pdf_pages_are_read_and_joined(
    app: FastAPI, client: TestClient, acct: Account
) -> None:
    pages = [TABLE[:2], [TABLE[0], *TABLE[2:3]]]  # page 2 repeats the header
    calls = iter(pages)

    class PerPage(FakeVision):
        def complete_with_image(self, prompt: str, image: bytes, mime: str, **kw: Any) -> Any:
            self.rows = next(calls)
            return super().complete_with_image(prompt, image, mime, **kw)

    vision = PerPage()
    app.dependency_overrides[get_vision_llm] = lambda: vision
    pdf = FPDF()
    pdf.add_page()
    pdf.add_page()  # two pages without a text layer, like a scan
    resp = upload(client, acct, "scan.pdf", bytes(pdf.output()))
    assert resp.status_code == 201, resp.text
    assert resp.json()["row_count"] == 2
    assert resp.json()["extraction"] == "ai_vision"
    assert [mime for _, mime in vision.images] == ["image/png", "image/png"]


def test_text_pdfs_never_call_the_vision_model(
    app: FastAPI, client: TestClient, acct: Account
) -> None:
    vision = FakeVision()
    app.dependency_overrides[get_vision_llm] = lambda: vision
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)
    with pdf.table() as table:
        for row in TABLE[:3]:
            r = table.row()
            for cell in row:
                r.cell(cell)
    resp = upload(client, acct, "report.pdf", bytes(pdf.output()))
    assert resp.status_code == 201, resp.text
    assert resp.json()["extraction"] is None
    assert vision.images == []


# --- providers --------------------------------------------------------------------------------


def test_gemini_sends_the_image_inline() -> None:
    seen: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen.update(json.loads(req.content))
        reply = json.dumps({"rows": TABLE})
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": reply}]}}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    rows, call = GeminiProvider("gemini-2.5-flash", "k", client).transcribe_table(
        b"img", "image/png"
    )
    parts = seen["contents"][0]["parts"]
    assert parts[0]["inline_data"] == {
        "mime_type": "image/png", "data": base64.b64encode(b"img").decode()
    }  # fmt: skip
    assert "Copy the table" in parts[1]["text"]
    assert seen["generationConfig"]["responseJsonSchema"]["required"] == ["rows"]
    assert rows == TABLE
    assert call.purpose == "vision"


def test_ollama_sends_images_field() -> None:
    seen: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen.update(json.loads(req.content))
        return httpx.Response(200, json={"message": {"content": json.dumps({"rows": TABLE})}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    rows, _ = OllamaProvider("qwen2.5vl:7b", "http://x", client).transcribe_table(
        b"img", "image/png"
    )
    assert seen["messages"][0]["images"] == [base64.b64encode(b"img").decode()]
    assert rows == TABLE


def test_text_only_providers_say_so() -> None:
    with pytest.raises(LLMUnavailableError, match="can't read images"):
        FakeLLM().transcribe_table(b"img", "image/png")


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({}, None),  # no key, nothing configured: images politely refused
        ({"gemini_api_key": "k"}, ("gemini", "gemini-2.5-flash")),  # sensible default
        ({"vision_provider": "ollama"}, ("ollama", "qwen2.5vl:7b")),
        ({"gemini_api_key": "k", "vision_provider": "none"}, None),
    ],
)
def test_vision_provider_selection(env: dict[str, Any], expected: Any) -> None:
    provider = make_vision_provider(Settings(_env_file=None, **env))
    assert (provider.name, provider.model) == expected if provider else expected is None


def test_ingest_rejects_images_when_no_transcriber(tmp_path: Path) -> None:
    from app.connectors.csv_xlsx import IngestError, ingest_file

    path = tmp_path / "x.png"
    path.write_bytes(png_bytes())
    with pytest.raises(IngestError, match="vision model"):
        ingest_file(path, "png", tmp_path / "out.parquet")
