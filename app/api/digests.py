"""Scheduled digests: create from an answer, manage, run now; and the hourly timer."""

import hmac
from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select

from app import audit
from app.api.deps import AppSettings, CurrentUser, DbSession, Today, not_found
from app.digests import MAX_ITEMS, describe_schedule, next_run, run_digest, run_due_digests, zone
from app.llm.base import LLMProvider
from app.llm.factory import get_llm
from app.models import Conversation, Dataset, Digest, DigestRun, Message, User

router = APIRouter(tags=["digests"])


class DigestIn(BaseModel):
    message_id: str  # the answer whose question is re-run on the schedule
    digest_id: str | None = None  # add it to this digest instead of making a new one
    title: str | None = Field(default=None, max_length=200)
    cadence: Literal["daily", "weekly"] = "weekly"
    weekday: int = Field(default=0, ge=0, le=6)  # 0 = Monday
    hour: int = Field(default=9, ge=0, le=23)
    minute: int = Field(default=0, ge=0, le=59)
    timezone: str | None = Field(default=None, max_length=64)
    recipients: list[EmailStr] | None = Field(default=None, max_length=20)


class DigestPatch(BaseModel):
    title: str | None = Field(default=None, max_length=200)
    cadence: Literal["daily", "weekly"] | None = None
    weekday: int | None = Field(default=None, ge=0, le=6)
    hour: int | None = Field(default=None, ge=0, le=23)
    minute: int | None = Field(default=None, ge=0, le=59)
    timezone: str | None = Field(default=None, max_length=64)
    recipients: list[EmailStr] | None = Field(default=None, max_length=20)
    active: bool | None = None
    remove_questions: list[str] | None = None


class DigestOut(BaseModel):
    id: str
    title: str
    dataset_id: str
    dataset_name: str
    questions: list[str]
    cadence: str
    weekday: int
    hour: int
    minute: int
    timezone: str
    schedule: str  # "every Monday at 09:00 (Asia/Karachi)"
    recipients: list[str]
    active: bool
    next_run_at: datetime
    last_run_at: datetime | None


class RunOut(BaseModel):
    id: str
    digest_id: str
    digest_title: str
    created_at: datetime
    dataset_version: int
    content: list[dict[str, Any]]
    changes: dict[str, Any] | None
    emailed: bool
    email_note: str | None


def _out(db: DbSession, d: Digest) -> DigestOut:
    ds = db.get(Dataset, d.dataset_id)
    return DigestOut(
        id=d.id, title=d.title, dataset_id=d.dataset_id, dataset_name=ds.name if ds else "",
        questions=[i["question"] for i in d.items], cadence=d.cadence, weekday=d.weekday,
        hour=d.hour, minute=d.minute, timezone=d.timezone, schedule=describe_schedule(d),
        recipients=list(d.recipients), active=d.active, next_run_at=d.next_run_at,
        last_run_at=d.last_run_at,
    )  # fmt: skip


def _run_out(db: DbSession, run: DigestRun) -> RunOut:
    d = db.get(Digest, run.digest_id)
    return RunOut(id=run.id, digest_id=run.digest_id, digest_title=d.title if d else "",
                  created_at=run.created_at, dataset_version=run.dataset_version,
                  content=run.content, changes=run.changes, emailed=run.emailed,
                  email_note=run.email_note)  # fmt: skip


def _own(db: DbSession, user: User, digest_id: str) -> Digest:
    d = db.get(Digest, digest_id)
    if d is None or d.user_id != user.id or d.workspace_id != user.workspace_id:
        raise not_found("Digest")
    return d


def answer_to_schedule(db: DbSession, user: User, message_id: str) -> tuple[Message, Message]:
    """The (question, answer) pair a digest re-runs; must be the user's computed answer."""
    reply = db.get(Message, message_id)
    if reply is None or reply.role != "assistant":
        raise not_found("Answer")
    conv = db.get(Conversation, reply.conversation_id)
    if conv is None or conv.user_id != user.id or conv.workspace_id != user.workspace_id:
        raise not_found("Answer")
    if not reply.plan or (reply.plan or {}).get("metric") == "describe":
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Only computed answers (numbers or tables) can be scheduled.",
        )
    question = next((m for m in conv.messages if m.position == reply.position - 1), None)
    if question is None:
        raise not_found("Answer")
    return question, reply


def create_digest(db: DbSession, user: User, body: DigestIn, default_tz: str) -> Digest:
    question, reply = answer_to_schedule(db, user, body.message_id)
    conv = db.get(Conversation, reply.conversation_id)
    if conv is None:
        raise not_found("Answer")
    item = {"question": question.content[:500], "plan": reply.plan}
    if body.digest_id:
        d = _own(db, user, body.digest_id)
        if d.dataset_id != conv.dataset_id:
            raise HTTPException(status.HTTP_400_BAD_REQUEST,
                                "That digest is about a different file.")  # fmt: skip
        if len(d.items) >= MAX_ITEMS:
            raise HTTPException(status.HTTP_400_BAD_REQUEST,
                                f"A digest holds at most {MAX_ITEMS} questions.")  # fmt: skip
        d.items = [*d.items, item]
        return d
    tz = body.timezone or default_tz
    zone(tz)  # unknown names fall back to UTC when scheduling
    d = Digest(
        workspace_id=user.workspace_id, user_id=user.id, dataset_id=conv.dataset_id,
        title=(body.title or question.content)[:200], items=[item], cadence=body.cadence,
        weekday=body.weekday, hour=body.hour, minute=body.minute, timezone=tz,
        recipients=[str(r) for r in body.recipients] if body.recipients else [user.email],
        active=True,
        next_run_at=next_run(body.cadence, body.weekday, body.hour, body.minute, tz,
                             datetime.now(UTC)),
    )  # fmt: skip
    db.add(d)
    db.flush()
    audit.record(db, user, "digest.created", target_type="digest", target_id=d.id)
    return d


@router.get("/digests", response_model=list[DigestOut])
def list_digests(user: CurrentUser, db: DbSession) -> list[DigestOut]:
    stmt = (select(Digest).where(Digest.user_id == user.id,
                                 Digest.workspace_id == user.workspace_id)
            .order_by(Digest.created_at.desc()))  # fmt: skip
    return [_out(db, d) for d in db.scalars(stmt)]


@router.post("/digests", response_model=DigestOut, status_code=status.HTTP_201_CREATED)
def add_digest(body: DigestIn, user: CurrentUser, db: DbSession,
               settings: AppSettings) -> DigestOut:  # fmt: skip
    d = create_digest(db, user, body, settings.default_timezone)
    db.commit()
    return _out(db, d)


@router.patch("/digests/{digest_id}", response_model=DigestOut)
def update_digest(digest_id: str, body: DigestPatch, user: CurrentUser,
                  db: DbSession) -> DigestOut:  # fmt: skip
    d = _own(db, user, digest_id)
    for name in ("title", "cadence", "weekday", "hour", "minute", "timezone", "active"):
        value = getattr(body, name)
        if value is not None:
            setattr(d, name, value)
    if body.recipients is not None:
        d.recipients = [str(r) for r in body.recipients]
    if body.remove_questions:
        d.items = [i for i in d.items if i["question"] not in body.remove_questions]
        if not d.items:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, "A digest needs at least one question; delete it."
            )
    d.next_run_at = next_run(d.cadence, d.weekday, d.hour, d.minute, d.timezone, datetime.now(UTC))
    db.commit()
    return _out(db, d)


@router.delete("/digests/{digest_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_digest(digest_id: str, user: CurrentUser, db: DbSession) -> None:
    d = _own(db, user, digest_id)
    audit.record(db, user, "digest.deleted", target_type="digest", target_id=d.id)
    db.delete(d)
    db.commit()


@router.post("/digests/{digest_id}/run", response_model=RunOut)
def run_now(digest_id: str, user: CurrentUser, db: DbSession, settings: AppSettings,
            today: Today, llm: Annotated[LLMProvider, Depends(get_llm)]) -> RunOut:  # fmt: skip
    """Send the digest now (it keeps its schedule)."""
    d = _own(db, user, digest_id)
    keep = d.next_run_at
    try:
        run = run_digest(db, settings, d, llm=llm, today=today)
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    d.next_run_at = keep
    db.commit()
    return _run_out(db, run)


@router.get("/digests/runs", response_model=list[RunOut])
def recent_runs(user: CurrentUser, db: DbSession,
                limit: Annotated[int, Query(ge=1, le=50)] = 20) -> list[RunOut]:  # fmt: skip
    """The latest digests sent, newest first (the in-app inbox)."""
    stmt = (select(DigestRun).join(Digest, Digest.id == DigestRun.digest_id)
            .where(Digest.user_id == user.id, Digest.workspace_id == user.workspace_id)
            .order_by(DigestRun.created_at.desc()).limit(limit))  # fmt: skip
    return [_run_out(db, r) for r in db.scalars(stmt)]


@router.post("/internal/digests/tick")
def digest_tick(
    db: DbSession,
    settings: AppSettings,
    llm: Annotated[LLMProvider, Depends(get_llm)],
    x_cron_secret: Annotated[str | None, Header()] = None,
) -> dict[str, int]:
    """Run every digest that's due. Called hourly by a timer (GitHub Actions) with the
    shared DIGEST_CRON_SECRET; disabled (404) when no secret is configured."""
    secret = settings.digest_cron_secret
    if secret is None:
        raise not_found("Endpoint")
    if not x_cron_secret or not hmac.compare_digest(x_cron_secret, secret.get_secret_value()):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Wrong timer secret.")
    runs = run_due_digests(db, settings, llm=llm)
    return {"sent": len(runs), "emailed": sum(1 for r in runs if r.emailed)}
