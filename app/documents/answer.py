"""Question about a document -> passages found by search -> cited, checked answer."""

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from app.documents.ingest import Passage, load_passages
from app.documents.search import search, tokenize
from app.llm.base import ChatMessage, LLMCall, LLMProvider
from app.llm.embeddings import Embedder
from app.query.presentation import text_is_grounded
from app.vectors import fuse, get_vectors, rank

TOP_K = 5
OVERVIEW_PASSAGES = 8
WHOLE_DOC_CHARS = 14_000  # documents up to ~4 pages are always read in full
# "Explain more", "tell me more", "elaborate", "go on": about the previous answer.
MORE = re.compile(
    r"^\W*(can you |could you |please )?(explain|tell( me)?|say|elaborate|go on|continue|"
    r"details?|more|expand)\b.{0,25}$",
    re.I,
)
# "What is this document about?", "summarize it", "main points": the whole document,
# not a keyword search (those words don't appear in the text itself).
OVERVIEW = re.compile(
    r"\b(what('?s|\s+is)\s+(this|the|it)\b.{0,30}\babout|summar(y|i[sz]e)|overview|"
    r"main\s+(points?|ideas?|topics?)|key\s+(points?|takeaways?)|gist|tl;?dr|"
    r"(explain|describe)\s+(this|the|it)\b(\s+(document|file|pdf|doc|paper|assignment))?\s*$)",
    re.I,
)


def spread(passages: list[Passage], k: int) -> list[Passage]:
    """Passages from across the whole document: the start, then evenly spaced."""
    if len(passages) <= k:
        return passages
    head = passages[:3]
    rest = passages[3:]
    step = len(rest) / (k - len(head))
    return head + [rest[int(i * step)] for i in range(k - len(head))]


Status = Literal["answered", "failed"]


@dataclass
class DocumentAnswer:
    status: Status
    text: str
    passages: list[Passage]  # what the user sees as sources (numbered 1..n)
    cited: list[int]  # which of them the answer used
    grounded: bool
    calls: list[LLMCall] = field(default_factory=list)


NOT_FOUND = (
    "I couldn't find anything about that in this document. Try other words, or ask about "
    "something the document covers."
)
NO_TEXT = (
    "I couldn't read any text in this document. If it's a scan or photos of pages, "
    "upload it as images (PNG/JPG) so AskData can read them."
)
UNGROUNDED = (
    "I couldn't give an answer I can back up with the document's exact words, so here are "
    "the passages that look most relevant."
)


def history_text(turns: list[tuple[str, str]]) -> list[ChatMessage]:
    msgs: list[ChatMessage] = []
    for question, answer in turns:
        msgs += [ChatMessage("user", f"Question: {question}"), ChatMessage("assistant", answer)]
    return msgs


def find_passages(
    query: str, passages: list[Passage], k: int, embedder: Embedder | None, parquet: Path
) -> list[Passage]:
    """Hybrid search: passages that share the query's words AND passages that mean the
    same thing in other words ("pay" for "salary"), merged. Keywords only when the
    embedding server isn't available."""
    keyword = search(query, passages, k=k * 2)
    vectors = get_vectors(embedder, parquet, "passages", [p.text for p in passages])
    if embedder is None or vectors is None:
        return keyword[:k]
    position = {p.chunk_id: i for i, p in enumerate(passages)}
    by_words = [position[p.chunk_id] for p in keyword]
    by_meaning = rank(embedder, vectors, query, k * 2)
    return [passages[i] for i in fuse(by_words, by_meaning, k=k)]


def answer_document_question(
    llm: LLMProvider,
    question: str,
    *,
    parquet: Path,
    history: list[tuple[str, str]],
    on_stage: Callable[[str], None] | None = None,
    embedder: Embedder | None = None,
) -> DocumentAnswer:
    """Raises LLMUnavailableError if the model can't be reached."""
    stage = on_stage or (lambda _s: None)
    stage("searching")
    passages = load_passages(parquet)
    if not passages:
        return DocumentAnswer("failed", NO_TEXT, [], [], grounded=True)
    previous_q, previous_a = history[-1] if history else ("", "")
    vague = len(tokenize(question)) < 2 or bool(MORE.search(question))
    # An overview question ("what is this about?", "summarise", or "explain more" after
    # one) looks across the whole document; a specific question gets only the passages
    # that actually answer it, never the entire document.
    is_overview = bool(OVERVIEW.search(question)) or (vague and bool(OVERVIEW.search(previous_q)))
    if is_overview:
        tiny = sum(len(p.text) for p in passages) <= WHOLE_DOC_CHARS
        found = passages if tiny else spread(passages, OVERVIEW_PASSAGES * (2 if vague else 1))
    else:
        # Follow-ups ("what about 2024?") search with the previous question/answer too.
        query = " ".join([previous_q, previous_a if vague else "", question])
        found = find_passages(query, passages, TOP_K * (2 if vague else 1), embedder, parquet)
        if not found:
            # No keyword overlap (different wording): a few passages from across the
            # document, not all of it. The model still says so if the answer isn't there.
            found = spread(passages, min(TOP_K, len(passages)))

    stage("reading")
    numbered = [(i, p.page, p.text) for i, p in enumerate(found, start=1)]
    answer, call = llm.answer_from_passages(question, numbered, history_text(history))
    if not answer.found or not answer.text:
        return DocumentAnswer("failed", NOT_FOUND, found, [], grounded=True, calls=[call])

    cited_texts = [found[n - 1].text for n in answer.sources] or [p.text for p in found]
    if not text_is_grounded(answer.text, cited_texts, question):
        # A number or date in the answer isn't in the passages it cites: don't show it.
        return DocumentAnswer("answered", UNGROUNDED, found, [], grounded=False, calls=[call])
    return DocumentAnswer("answered", answer.text, found, answer.sources, True, calls=[call])
