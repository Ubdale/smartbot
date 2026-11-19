"""Search by meaning: embeddings client, vector cache, hybrid search for documents and
table topics. A fake embedder that knows a few synonym groups stands in for the model."""

import json
from pathlib import Path

import duckdb
import httpx
import numpy as np
import pytest

from app.documents.answer import find_passages
from app.documents.ingest import Passage
from app.llm.embeddings import Embedder, EmbeddingError, Vectors, get_embedder
from app.vectors import fuse, get_vectors

GROUPS = [
    {"salary", "salaries", "pay", "paid", "wage", "wages", "compensation"},
    {"leave", "holiday", "holidays", "vacation", "annual"},
    {"tax", "vat", "withholding", "duty"},
    {"login", "password", "sign-in", "authentication"},
]


class ConceptEmbedder(Embedder):
    """Vectors from synonym groups: texts sharing a meaning point the same way."""

    def __init__(self) -> None:
        super().__init__("http://unused", "fake")
        self.calls = 0

    def embed(self, texts: list[str], *, query: bool = False) -> Vectors:
        self.calls += 1
        rows = []
        for t in texts:
            words = {w.strip(".,?!()").lower() for w in t.split()}
            v = [float(len(words & g)) for g in GROUPS] + [0.1]
            rows.append(v)
        m = np.asarray(rows, dtype=np.float32)
        out: Vectors = (m / np.linalg.norm(m, axis=1, keepdims=True)).astype(np.float32)
        return out


def test_embedder_calls_ollama_and_pauses_after_failure() -> None:
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        assert request.url.path == "/api/embed"
        assert request.headers["authorization"] == "Bearer tok"
        return httpx.Response(200, json={"embeddings": [[3.0, 4.0] for _ in body["input"]]})

    client = httpx.Client(transport=httpx.MockTransport(handler),
                          headers={"Authorization": "Bearer tok"})  # fmt: skip
    e = Embedder("http://box", "nomic-embed-text", client=client)
    v = e.embed(["a", "b"], query=True)
    assert v.shape == (2, 2)
    assert np.allclose(v[0], [0.6, 0.8])  # normalised
    assert seen[0]["input"] == ["search_query: a", "search_query: b"]

    broken = Embedder("http://box", "m", client=httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(500))))  # fmt: skip
    with pytest.raises(EmbeddingError):
        broken.embed(["x"])
    assert not broken.available  # paused: answers don't keep waiting on a dead server


def test_get_embedder_is_off_in_tests_and_when_not_ollama(settings: object) -> None:
    from app.core.config import Settings

    assert isinstance(settings, Settings)
    assert get_embedder(settings) is None  # app_env=test
    on = settings.model_copy(update={"app_env": "prod", "llm_provider": "ollama"})
    assert get_embedder(on) is not None
    groq = settings.model_copy(update={"app_env": "prod", "llm_provider": "groq"})
    assert get_embedder(groq) is None


def test_vectors_are_cached_on_disk(tmp_path: Path) -> None:
    parquet = tmp_path / "data.parquet"
    parquet.write_bytes(b"x")
    e = ConceptEmbedder()
    texts = ["pay rise", "holiday rules"]
    first = get_vectors(e, parquet, "passages", texts)
    assert first is not None and e.calls == 1
    assert get_vectors(e, parquet, "passages", texts) is not None
    assert e.calls == 1  # served from cache
    assert list(tmp_path.glob("*.vec.npy"))
    assert get_vectors(None, parquet, "passages", texts) is None


def test_fuse_merges_rankings() -> None:
    assert fuse([1, 2, 3], [3, 1, 4], k=3) == [1, 3, 2]


def test_documents_find_passages_with_other_words(tmp_path: Path) -> None:
    parquet = tmp_path / "doc.parquet"
    parquet.write_bytes(b"x")
    passages = [Passage(i, None, t) for i, t in enumerate([
        "Our office is in Lahore and opens at 9.",
        "Employees are paid on the last working day; wages include overtime.",
        "Annual holiday allowance is 24 days.",
        "Parking is free for staff.",
    ] + [f"Unrelated policy number {n}." for n in range(20)])]  # fmt: skip
    question = "when do we get our salary?"
    # Keywords alone miss it: "salary" never appears in the document.
    assert all("paid" not in p.text for p in find_passages(question, passages, 3, None, parquet))
    found = find_passages(question, passages, 3, ConceptEmbedder(), parquet)
    assert any("paid" in p.text for p in found)


def test_table_topic_shortlist_uses_meaning(tmp_path: Path) -> None:
    import pandas as pd

    from app.query.retrieval import match_topic
    from tests.fakes import FakeLLM

    texts = [f"Update dashboard widget {i}" for i in range(80)] + [
        "VAT return mismatch in October",  # no word "tax"
        "Withholding report wrong totals",
    ]
    parquet = tmp_path / "t.parquet"
    frame = pd.DataFrame({"Summary": texts})
    with duckdb.connect() as con:
        con.register("frame", frame)
        con.execute(f"COPY frame TO '{parquet.as_posix()}' (FORMAT PARQUET)")

    class Checker(FakeLLM):
        def complete(self, messages, *, json_schema=None, max_tokens=512):  # type: ignore[no-untyped-def]
            listing = messages[-1].content
            picked = [i for i, line in enumerate(listing.split("\n")[3:], start=1)
                      if "VAT" in line or "Withholding" in line]  # fmt: skip
            from app.llm.base import Completion

            return Completion(json.dumps({"matches": picked}), 10, 5, 1)

    m = match_topic(Checker(), parquet, "Summary", "tax problems", ConceptEmbedder())
    assert sorted(m.rows) == [80, 81]
    assert m.checked <= 60  # a shortlist, not all 82 texts
