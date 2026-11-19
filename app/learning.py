"""How AskData learns from use, without retraining any model.

- Worked examples: questions whose answers users marked helpful are shown to the model
  as examples when someone asks something similar about the same file.
- Vocabulary: "sales means revenue" is remembered per file and given to the model.
- Instant repeats: an identical opening question reuses the earlier (unrated or
  helpful) plan, so no model call is needed; the numbers are always recomputed.
"""

import re
from dataclasses import dataclass
from datetime import date
from typing import Any

from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session, aliased

from app.documents.search import tokenize
from app.models import Conversation, Dataset, GlossaryTerm, LearnedExample, Message
from app.query.executor import PlanError, columns_from_profile, validate_plan
from app.schemas.query_plan import QueryPlan

MAX_EXAMPLES = 3
MIN_SIMILARITY = 0.25
MAX_TERMS = 30


def question_key(question: str) -> str:
    """Case, punctuation and spacing don't make a different question."""
    return re.sub(r"[^a-z0-9]+", " ", question.lower()).strip()[:500]


# --- vocabulary ---------------------------------------------------------------------------

_DEFINITION = [
    re.compile(
        r"^\s*(?:by\s+|when\s+i\s+say\s+)?[\"'“]?(?P<term>[\w][\w\s\-/&]{0,59}?)[\"'”]?\s+"
        r"(?:means|=|is\s+the\s+same\s+as|refers\s+to|stands\s+for|i\s+mean)\s+"
        r"[\"'“]?(?P<meaning>[^\"'”?]{1,200}?)[\"'”]?\s*[.!]?\s*$",
        re.I,
    ),
]


# "this means…", "it means…", "what means…" are sentences, not vocabulary.
_NOT_A_TERM = frozenset(
    [
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "which",
        "what",
        "there",
        "here",
        "he",
        "she",
        "they",
        "we",
        "i",
        "you",
        "my",
        "our",
        "everything",
        "nothing",
        "something",
    ]
)


@dataclass(frozen=True)
class Definition:
    term: str
    meaning: str


def parse_definition(text: str) -> Definition | None:
    """'sales means revenue' / 'by staff I mean headcount' / 'GP = gross profit'."""
    if "?" in text or len(text) > 250:
        return None
    for pattern in _DEFINITION:
        if m := pattern.match(text):
            term = " ".join(m["term"].split())
            meaning = " ".join(m["meaning"].split())
            first_word = term.split()[0].lower() if term else ""
            if (
                term
                and meaning
                and len(term.split()) <= 4
                and len(meaning.split()) <= 8  # definitions are short; chat sentences aren't
                and first_word not in _NOT_A_TERM
                and term.lower() != meaning.lower()
            ):
                return Definition(term, meaning)
    return None


def remember_term(
    db: Session, *, workspace_id: str, dataset_id: str, user_id: str, d: Definition
) -> None:
    existing = db.scalar(
        select(GlossaryTerm).where(
            GlossaryTerm.workspace_id == workspace_id,
            GlossaryTerm.dataset_id == dataset_id,
            func.lower(GlossaryTerm.term) == d.term.lower(),
        )
    )
    if existing:
        existing.meaning = d.meaning
    else:
        db.add(
            GlossaryTerm(
                workspace_id=workspace_id,
                dataset_id=dataset_id,
                term=d.term,
                meaning=d.meaning,
                created_by_id=user_id,
            )
        )


def glossary_notes(db: Session, *, workspace_id: str, dataset_id: str) -> str:
    """Team vocabulary for the model, as a line under the table description."""
    terms = db.scalars(
        select(GlossaryTerm)
        .where(GlossaryTerm.workspace_id == workspace_id, GlossaryTerm.dataset_id == dataset_id)
        .order_by(GlossaryTerm.created_at.desc())
        .limit(MAX_TERMS)
    ).all()
    if not terms:
        return ""
    pairs = "; ".join(f'"{t.term}" means {t.meaning}' for t in terms)
    return f"Words this team uses: {pairs}."


# --- worked examples ----------------------------------------------------------------------


def _similarity(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def similar_examples(
    db: Session, *, workspace_id: str, ds: Dataset, question: str, today: date
) -> list[tuple[str, dict[str, Any]]]:
    """Up to 3 helpful past (question, plan) pairs most like this question, still valid."""
    rows = db.scalars(
        select(LearnedExample)
        .where(LearnedExample.workspace_id == workspace_id, LearnedExample.dataset_id == ds.id)
        .order_by(LearnedExample.created_at.desc())
        .limit(200)
    ).all()
    words = set(tokenize(question))
    columns = columns_from_profile(ds.columns)
    scored = sorted(
        ((_similarity(words, set(tokenize(r.question))), r) for r in rows),
        key=lambda pair: -pair[0],
    )
    picked: list[tuple[str, dict[str, Any]]] = []
    for score, example in scored:
        if score < MIN_SIMILARITY or len(picked) >= MAX_EXAMPLES:
            break
        try:  # the file may have changed since: only show plans that still fit it
            validate_plan(QueryPlan.model_validate(example.plan), columns, today)
        except (PlanError, ValueError):
            continue
        picked.append((example.question, example.plan))
    return picked


def learn_from_rating(
    db: Session, *, reply: Message, question: Message, conv: Conversation, user_id: str
) -> None:
    """👍 on a computed answer saves it as an example; 👎 (or undo) removes it."""
    key = question_key(question.content)
    existing = db.scalars(
        select(LearnedExample).where(
            LearnedExample.workspace_id == conv.workspace_id,
            LearnedExample.dataset_id == conv.dataset_id,
            LearnedExample.question_key == key,
        )
    ).all()
    if reply.rating == 1 and reply.plan:
        if existing:
            existing[0].plan = reply.plan
            return
        db.add(
            LearnedExample(
                workspace_id=conv.workspace_id,
                dataset_id=conv.dataset_id,
                question=question.content,
                question_key=key,
                plan=reply.plan,
                created_by_id=user_id,
            )
        )
    elif reply.rating == -1:
        for example in existing:
            db.delete(example)


# --- instant repeats ----------------------------------------------------------------------


@dataclass(frozen=True)
class Remembered:
    plan: QueryPlan
    summary: str
    rows: list[list[Any]]
    summary_source: str


def remembered_answer(
    db: Session, *, workspace_id: str, dataset_id: str, question: str
) -> Remembered | None:
    """An earlier answer to the same opening question about this file (not rated wrong)."""
    asked = aliased(Message)
    stmt = (
        select(Message)
        .join(Conversation, Conversation.id == Message.conversation_id)
        .join(
            asked,
            and_(asked.conversation_id == Message.conversation_id, asked.position == 0),
        )
        .where(
            Conversation.workspace_id == workspace_id,
            Conversation.dataset_id == dataset_id,
            Message.position == 1,
            Message.role == "assistant",
            Message.status == "answered",
            Message.plan.is_not(None),
            or_(Message.rating.is_(None), Message.rating == 1),
            func.lower(func.trim(asked.content)) == question.strip().lower(),
        )
        .order_by(Message.rating.desc().nulls_last(), Message.created_at.desc())
        .limit(5)
    )
    for reply in db.scalars(stmt):
        if not reply.plan or not reply.answer or reply.answer.get("type", "table") != "table":
            continue
        try:
            plan = QueryPlan.model_validate(reply.plan)
        except ValueError:
            continue
        return Remembered(
            plan=plan,
            summary=reply.content,
            rows=reply.answer.get("table", {}).get("rows", []),
            summary_source=str(reply.answer.get("summary_source", "template")),
        )
    return None
