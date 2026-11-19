"""Ask questions about a dataset; list and read conversations."""

from datetime import UTC, datetime
from itertools import pairwise
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select

from app.api.datasets import require_parquet
from app.api.deps import AppSettings, CurrentUser, DbSession, Today, not_found, workspace_scoped
from app.documents.answer import DocumentAnswer, answer_document_question
from app.llm.base import LLMCall, LLMProvider, LLMUnavailableError
from app.llm.factory import get_llm, get_summary_llm
from app.models import Conversation, Dataset, Message, UsageEvent, User
from app.query.answer import Answer, answer_question
from app.query.planner import history_messages
from app.schemas.chat import (
    AskRequest,
    AskResponse,
    ConversationOut,
    ConversationSummary,
    MessageOut,
)

router = APIRouter(prefix="/chat", tags=["chat"])

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
                if q.role == "user" and a.role == "assistant" and a.status == "answered"
            ]
            doc = answer_document_question(
                llm, body.question, parquet=parquet, history=doc_turns[-2:]
            )
            content, status_, plan, payload, calls = (
                doc.text, doc.status, None, _document_payload(doc), doc.calls
            )  # fmt: skip
        else:
            # Recent answered turns let follow-ups ("now split by city") modify the plan.
            turns = [
                (q.content, a.plan)
                for q, a in previous
                if q.role == "user" and a.role == "assistant" and a.plan
            ]
            result = answer_question(
                llm,
                body.question,
                dataset_id=ds.id,
                dataset_name=ds.name,
                dataset_version=ds.version,
                row_count=ds.row_count,
                profile=ds.columns,
                parquet=parquet,
                history=history_messages(turns[-HISTORY_TURNS:]),
                today=today,
                summarizer=summarizer,
            )
            content, status_, payload, calls = (
                result.summary, result.status, _answer_payload(result, ds), result.calls
            )  # fmt: skip
            plan = result.validated.plan.model_dump(mode="json") if result.validated else None
    except LLMUnavailableError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc

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
