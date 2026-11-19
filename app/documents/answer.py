"""Question about a document -> passages found by search -> cited, checked answer."""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from app.documents.ingest import Passage, load_passages
from app.documents.search import search
from app.llm.base import ChatMessage, LLMCall, LLMProvider
from app.query.presentation import text_is_grounded

TOP_K = 5

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
    found = search(query, passages, k=TOP_K)
    if not found:
        return DocumentAnswer("failed", NOT_FOUND, [], [], grounded=True)

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
