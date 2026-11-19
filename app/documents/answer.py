"""Question about a document -> passages found by search -> cited, checked answer."""

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from app.documents.ingest import Passage, load_passages
from app.documents.search import search
from app.llm.base import ChatMessage, LLMCall, LLMProvider
from app.query.presentation import text_is_grounded

TOP_K = 5
OVERVIEW_PASSAGES = 8
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


def answer_document_question(
    llm: LLMProvider,
    question: str,
    *,
    parquet: Path,
    history: list[tuple[str, str]],
    on_stage: Callable[[str], None] | None = None,
) -> DocumentAnswer:
    """Raises LLMUnavailableError if the model can't be reached."""
    stage = on_stage or (lambda _s: None)
    stage("searching")
    passages = load_passages(parquet)
    # Follow-ups ("what about 2024?") search with the previous question's words too.
    query = " ".join([q for q, _ in history[-1:]] + [question])
    if not passages:
        return DocumentAnswer("failed", NO_TEXT, [], [], grounded=True)
    if OVERVIEW.search(question):
        found = spread(passages, OVERVIEW_PASSAGES)
    else:
        found = search(query, passages, k=TOP_K)
        if not found:
            # No shared keywords (different wording): let the model read the beginning
            # rather than giving up. It still says so if the answer isn't there.
            found = spread(passages, TOP_K)

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
