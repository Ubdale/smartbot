"""Ask questions about a dataset; list and read conversations."""

import json
import logging
import queue
import re
import threading
from collections.abc import Callable, Iterator
from datetime import UTC, date, datetime
from itertools import pairwise
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app import learning
from app.api.datasets import require_parquet
from app.api.deps import AppSettings, CurrentUser, DbSession, Today, not_found, workspace_scoped
from app.core import ratelimit
from app.core.config import Settings
from app.core.db import get_session_factory
from app.documents.answer import DocumentAnswer, answer_document_question
from app.llm.base import LLMCall, LLMProvider, LLMUnavailableError
from app.llm.factory import get_llm, get_summary_llm
from app.models import Conversation, Dataset, Message, UsageEvent, User
from app.query.answer import Answer, answer_question
from app.query.codegen import pandas_code
from app.query.planner import history_messages
from app.schemas.chat import (
    AskRequest,
    AskResponse,
    ConversationOut,
    ConversationSummary,
    FeedbackOut,
    FeedbackRequest,
    MessageOut,
)

router = APIRouter(prefix="/chat", tags=["chat"])
log = logging.getLogger(__name__)

HISTORY_TURNS = 3


def _conversation(db: DbSession, user: User, conversation_id: str) -> Conversation:
    stmt = workspace_scoped(
        select(Conversation).where(
            Conversation.id == conversation_id, Conversation.user_id == user.id
        ),
        Conversation,
        user,
    )
    conv = db.scalar(stmt)
    if conv is None:
        raise not_found("Conversation")
    return conv


def _dataset(db: DbSession, user: User, dataset_id: str) -> Dataset:
    stmt = workspace_scoped(select(Dataset).where(Dataset.id == dataset_id), Dataset, user)
    ds = db.scalar(stmt)
    if ds is None:
        raise not_found("Dataset")
    return ds


AI_VISION_NOTE = (
    "This table was read from an image by AI. The calculation is exact, but check the "
    "dataset preview to make sure the numbers were read correctly."
)


def _answer_payload(a: Answer, ds: Dataset) -> dict[str, Any] | None:
    payload = _table_payload(a, ds)
    if payload is not None:
        payload["reused"] = a.reused
    return payload


def _table_payload(a: Answer, ds: Dataset) -> dict[str, Any] | None:
    if a.status != "answered" or a.result is None or a.validated is None:
        return None
    r, vp = a.result, a.validated
    dr = vp.date_range
    return {
        "table": {"columns": r.columns, "rows": r.rows, "truncated": r.truncated},
        "chart": a.chart.__dict__ if a.chart else None,
        "explanation": {
            "description": a.description or "",
            "sql": r.sql,
            "params": r.params,
            "date_range": (
                {"start": dr.start.isoformat(), "end": dr.last_day.isoformat(), "label": dr.label}
                if dr
                else None
            ),
            "plan": vp.plan.model_dump(mode="json"),
            "cached": r.cached,
            "elapsed_ms": r.elapsed_ms,
            "data_note": AI_VISION_NOTE if ds.extraction == "ai_vision" else None,
        },
        "summary_source": a.summary_source,
    }


def _document_payload(doc: DocumentAnswer) -> dict[str, Any] | None:
    if not doc.passages:
        return None
    return {
        "type": "document",
        "grounded": doc.grounded,
        "sources": [
            {"n": i, "page": p.page, "text": p.text, "cited": i in doc.cited}
            for i, p in enumerate(doc.passages, start=1)
        ],
    }


@router.post("/ask", response_model=AskResponse)
def ask(
    body: AskRequest,
    user: CurrentUser,
    db: DbSession,
    settings: AppSettings,
    llm: Annotated[LLMProvider, Depends(get_llm)],
    summarizer: Annotated[LLMProvider, Depends(get_summary_llm)],
    today: Today,
) -> AskResponse:
    ratelimit.questions.check(user.id, settings.questions_per_minute)
    return run_ask(body, user, db, settings, llm, summarizer, today)


@router.post("/ask/stream")
def ask_stream(
    body: AskRequest,
    user: CurrentUser,
    settings: AppSettings,
    llm: Annotated[LLMProvider, Depends(get_llm)],
    summarizer: Annotated[LLMProvider, Depends(get_summary_llm)],
    today: Today,
    sessions: Annotated[sessionmaker[Session], Depends(get_session_factory)],
) -> StreamingResponse:
    """Same as /ask, as server-sent events: progress stages first, then the answer.

    Events: {"type": "stage", "stage": "planning"|"querying"|"summarizing"|"searching"|
    "reading"}, then {"type": "done", "response": AskResponse} or {"type": "error",
    "status": int, "detail": str}.
    """
    ratelimit.questions.check(user.id, settings.questions_per_minute)
    events: queue.Queue[dict[str, Any] | None] = queue.Queue()
    user_id = user.id

    def work() -> None:
        with sessions() as db:  # its own session: it outlives the request's
            try:
                worker_user = db.get(User, user_id)
                if worker_user is None:
                    raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not authenticated")
                response = run_ask(
                    body, worker_user, db, settings, llm, summarizer, today,
                    on_stage=lambda stage: events.put({"type": "stage", "stage": stage}),
                )  # fmt: skip
                events.put({"type": "done", "response": response.model_dump(mode="json")})
            except HTTPException as exc:
                events.put({"type": "error", "status": exc.status_code, "detail": exc.detail})
            except Exception:
                log.exception("streamed ask failed")
                events.put({"type": "error", "status": 500, "detail": "Something went wrong."})
            finally:
                events.put(None)

    threading.Thread(target=work, daemon=True).start()

    def stream() -> Iterator[str]:
        yield ": start\n\n"  # flush headers through proxies right away
        while (event := events.get()) is not None:
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )


def run_ask(
    body: AskRequest,
    user: User,
    db: Session,
    settings: Settings,
    llm: LLMProvider,
    summarizer: LLMProvider,
    today: date,
    on_stage: Callable[[str], None] | None = None,
) -> AskResponse:
    if body.conversation_id:
        conv = _conversation(db, user, body.conversation_id)
        ds = _dataset(db, user, conv.dataset_id)
    elif body.dataset_id:
        ds = _dataset(db, user, body.dataset_id)
        conv = Conversation(
            workspace_id=user.workspace_id,
            user_id=user.id,
            dataset_id=ds.id,
            title=body.question[:200],
        )
    else:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, "Send dataset_id or conversation_id."
        )

    parquet = require_parquet(settings, ds)
    previous = list(pairwise(conv.messages)) if body.conversation_id else []

    # Requests about the conversation itself, answered directly (no model call).
    if CODE_REQUEST.search(body.question) and ds.kind == "table":
        return _code_reply(db, conv, body, ds, previous, user)
    if EXPORT_REQUEST.search(body.question):
        return _save_turn(
            db, conv, body,
            content=(
                "You can download any answer: use the ⬇ CSV button under its table and "
                "⬇ PNG under its chart. Ask for “Python code” to get code that "
                "recreates the latest answer."
            ),
            status_="answered", plan=None, payload=None, calls=[], user=user,
        )  # fmt: skip

    definition = learning.parse_definition(body.question)
    if definition is not None:
        # "sales means revenue": remember it for this file. No model needed.
        learning.remember_term(
            db, workspace_id=user.workspace_id, dataset_id=ds.id, user_id=user.id, d=definition
        )
        return _save_turn(
            db,
            conv,
            body,
            content=(
                f"Got it. From now on, “{definition.term}” means "
                f"{definition.meaning} for “{ds.name}”."
            ),
            status_="answered",
            plan=None,
            payload=None,
            calls=[],
            user=user,
        )

    content: str
    status_: str
    plan: dict[str, Any] | None
    payload: dict[str, Any] | None
    calls: list[LLMCall]
    try:
        if ds.kind == "document":
            # Earlier questions and answers give follow-ups their context.
            doc_turns = [
                (q.content, a.content)
                for q, a in previous
                if q.role == "user" and a.role == "assistant" and a.status != "failed"
            ]
            doc = answer_document_question(
                llm,
                body.question,
                parquet=parquet,
                history=doc_turns[-2:],
                on_stage=on_stage,
            )
            content, status_, plan, payload, calls = (
                doc.text, doc.status, None, _document_payload(doc), doc.calls
            )  # fmt: skip
        else:
            # Recent answered turns let follow-ups ("now split by city") modify the plan.
            turns: list[tuple[str, dict[str, Any]]] = []
            for q, a in previous:
                if q.role != "user" or a.role != "assistant":
                    continue
                if a.plan:
                    turns.append((q.content, a.plan))
                elif a.status == "clarification":  # so "Yes" means yes to *this*
                    turns.append((q.content, {"clarification_needed": a.content}))
            # What AskData has learned about this file: helpful past answers to similar
            # questions (as examples), and the team's words for things.
            examples = learning.similar_examples(
                db, workspace_id=user.workspace_id, ds=ds, question=body.question, today=today
            )
            remembered = (
                None
                if turns  # follow-ups depend on context: never reuse those
                else learning.remembered_answer(
                    db, workspace_id=user.workspace_id, dataset_id=ds.id, question=body.question
                )
            )
            result = answer_question(
                llm,
                body.question,
                dataset_id=ds.id,
                dataset_name=ds.name,
                dataset_version=ds.version,
                row_count=ds.row_count,
                profile=ds.columns,
                parquet=parquet,
                history=history_messages(examples + turns[-HISTORY_TURNS:]),
                today=today,
                summarizer=summarizer,
                on_stage=on_stage,
                notes=learning.glossary_notes(db, workspace_id=user.workspace_id, dataset_id=ds.id),
                remembered=remembered,
            )
            content, status_, payload, calls = (
                result.summary, result.status, _answer_payload(result, ds), result.calls
            )  # fmt: skip
            plan = result.validated.plan.model_dump(mode="json") if result.validated else None
    except LLMUnavailableError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc

    return _save_turn(
        db, conv, body, content=content, status_=status_, plan=plan, payload=payload,
        calls=calls, user=user,
    )  # fmt: skip


CODE_REQUEST = re.compile(
    r"\b(python|pandas|sql|script|jupyter|notebook)\b|\b(the|this|some|write|in|give)\s+code\b",
    re.I,
)
EXPORT_REQUEST = re.compile(r"\b(download|export)\b", re.I)


def _code_reply(
    db: Session,
    conv: Conversation,
    body: AskRequest,
    ds: Dataset,
    previous: list[tuple[Message, Message]],
    user: User,
) -> AskResponse:
    """Code that reproduces the latest computed answer, generated from its plan."""
    last = next(
        (
            (q, a)
            for q, a in reversed(previous)
            if q.role == "user" and a.role == "assistant" and a.plan and a.answer
        ),
        None,
    )
    if last is None:
        return _save_turn(
            db, conv, body,
            content=(
                "Ask me a question about the data first (for example “top 10 customers "
                "by revenue”), then ask for Python code and I'll give you code that "
                "reproduces that answer."
            ),
            status_="answered", plan=None, payload=None, calls=[], user=user,
        )  # fmt: skip
    question, answer = last
    explanation = (answer.answer or {}).get("explanation", {})
    python = pandas_code(
        answer.plan or {},
        file_name=ds.original_filename,
        file_format=ds.file_format,
        date_range=explanation.get("date_range"),
    )
    blocks = [{"language": "python", "code": python}]
    if explanation.get("sql", "").startswith("SELECT"):
        params = ", ".join(repr(p) for p in explanation.get("params", []))
        blocks.append(
            {
                "language": "sql",
                "code": f"-- DuckDB; parameters in order: {params}\n"
                + explanation["sql"].replace("read_parquet(?)", repr(ds.original_filename)),
            }
        )
    return _save_turn(
        db, conv, body,
        content=(
            f"Here's Python (pandas) code that reproduces “{question.content}”. "
            f"Put {ds.original_filename} next to it, or change the file path."
        ),
        status_="answered", plan=None, payload={"type": "code", "blocks": blocks},
        calls=[], user=user,
    )  # fmt: skip


def _save_turn(
    db: Session,
    conv: Conversation,
    body: AskRequest,
    *,
    content: str,
    status_: str,
    plan: dict[str, Any] | None,
    payload: dict[str, Any] | None,
    calls: list[LLMCall],
    user: User,
) -> AskResponse:
    now = datetime.now(UTC)
    position = len(conv.messages) if body.conversation_id else 0
    question = Message(role="user", content=body.question, position=position, created_at=now)
    reply = Message(
        role="assistant",
        content=content,
        status=status_,
        plan=plan,
        answer=payload,
        position=position + 1,
        created_at=now,
    )
    conv.messages.extend([question, reply])
    conv.updated_at = now
    db.add(conv)
    for call in calls:
        db.add(
            UsageEvent(
                workspace_id=user.workspace_id,
                user_id=user.id,
                purpose=call.purpose,
                provider=call.provider,
                model=call.model,
                prompt_tokens=call.prompt_tokens,
                completion_tokens=call.completion_tokens,
                latency_ms=call.latency_ms,
                success=call.success,
            )
        )
    db.commit()
    return AskResponse(
        conversation_id=conv.id,
        question=MessageOut.model_validate(question),
        reply=MessageOut.model_validate(reply),
    )


@router.get("/conversations", response_model=list[ConversationSummary])
def list_conversations(user: CurrentUser, db: DbSession) -> list[Conversation]:
    stmt = (
        workspace_scoped(select(Conversation), Conversation, user)
        .where(Conversation.user_id == user.id)
        .order_by(Conversation.updated_at.desc())
    )
    return list(db.scalars(stmt))


@router.get("/conversations/{conversation_id}", response_model=ConversationOut)
def get_conversation(conversation_id: str, user: CurrentUser, db: DbSession) -> Conversation:
    return _conversation(db, user, conversation_id)


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(conversation_id: str, user: CurrentUser, db: DbSession) -> None:
    db.delete(_conversation(db, user, conversation_id))
    db.commit()


@router.post("/messages/{message_id}/feedback", response_model=FeedbackOut)
def give_feedback(
    message_id: str, body: FeedbackRequest, user: CurrentUser, db: DbSession
) -> FeedbackOut:
    """👍 teaches AskData this answer was right (it becomes an example for similar
    questions); 👎 makes sure it's never reused or learned from."""
    reply = db.get(Message, message_id)
    if reply is None or reply.role != "assistant":
        raise not_found("Message")
    conv = _conversation(db, user, reply.conversation_id)  # also checks ownership
    question = next((m for m in conv.messages if m.position == reply.position - 1), None)
    reply.rating = body.rating or None
    if question is not None:
        learning.learn_from_rating(db, reply=reply, question=question, conv=conv, user_id=user.id)
    db.commit()
    return FeedbackOut(
        message_id=reply.id,
        rating=reply.rating,
        learned=reply.rating == 1 and reply.plan is not None,
    )
