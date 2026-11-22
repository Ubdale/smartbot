"""Team management: invite people by link, roles, removal, and the audit log.

Roles: owner (everything, incl. promoting owners), admin (invite/remove members,
see the audit log), member (use the data). A workspace always keeps an owner.
"""

import hashlib
import secrets
from datetime import UTC, datetime, timedelta
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, EmailStr, field_validator
from sqlalchemy import func, select

from app import audit
from app.api.deps import AdminUser, CurrentUser, DbSession, not_found
from app.models import AuditEvent, Conversation, Dataset, Invite, Message, User
from app.schemas.auth import UserOut

router = APIRouter(prefix="/workspaces/me", tags=["team"])

INVITE_DAYS = 7


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class InviteIn(BaseModel):
    email: EmailStr
    role: Literal["admin", "member"] = "member"

    @field_validator("email")
    @classmethod
    def _lower(cls, v: str) -> str:
        return v.lower()


class ActivityOut(BaseModel):
    questions_total: int
    questions_this_week: int
    datasets: int
    conversations: int
    active_members: int
    top_datasets: list[dict[str, object]]  # [{name, questions}]
    busiest_people: list[dict[str, object]]  # [{email, questions}]


class InviteOut(BaseModel):
    id: str
    email: str
    role: str
    expires_at: datetime
    created_at: datetime


class InviteCreated(InviteOut):
    token: str  # shown once; the link is /invite/{token}


class RoleIn(BaseModel):
    role: Literal["owner", "admin", "member"]


class AuditOut(BaseModel):
    id: str
    actor_email: str | None
    action: str
    target_type: str | None
    target_id: str | None
    details: dict[str, object] | None
    created_at: datetime


def _owners(db: DbSession, workspace_id: str) -> int:
    return int(
        db.scalar(
            select(func.count())
            .select_from(User)
            .where(User.workspace_id == workspace_id, User.role == "owner")
        )
        or 0
    )


def _member(db: DbSession, actor: User, user_id: str) -> User:
    member = db.get(User, user_id)
    if member is None or member.workspace_id != actor.workspace_id:
        raise not_found("Member")
    return member


@router.post("/invites", response_model=InviteCreated, status_code=status.HTTP_201_CREATED)
def create_invite(body: InviteIn, actor: AdminUser, db: DbSession) -> InviteCreated:
    if db.scalar(select(User.id).where(User.email == body.email)) is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "That email already has an AskData account.")
    token = secrets.token_urlsafe(32)
    invite = Invite(
        workspace_id=actor.workspace_id,
        email=body.email,
        role=body.role,
        token_hash=hash_token(token),
        created_by_id=actor.id,
        expires_at=datetime.now(UTC) + timedelta(days=INVITE_DAYS),
    )
    db.add(invite)
    audit.record(db, actor, "invite.created", target_type="invite", email=body.email,
                 role=body.role)  # fmt: skip
    db.commit()
    return InviteCreated(
        id=invite.id,
        email=invite.email,
        role=invite.role,
        expires_at=invite.expires_at,
        created_at=invite.created_at,
        token=token,
    )


@router.get("/invites", response_model=list[InviteOut])
def list_invites(actor: AdminUser, db: DbSession) -> list[Invite]:
    stmt = select(Invite).where(
        Invite.workspace_id == actor.workspace_id,
        Invite.accepted_at.is_(None),
        Invite.expires_at > datetime.now(UTC),
    )
    return list(db.scalars(stmt.order_by(Invite.created_at.desc())))


@router.delete("/invites/{invite_id}", status_code=status.HTTP_204_NO_CONTENT)
def revoke_invite(invite_id: str, actor: AdminUser, db: DbSession) -> None:
    invite = db.get(Invite, invite_id)
    if invite is None or invite.workspace_id != actor.workspace_id:
        raise not_found("Invite")
    audit.record(db, actor, "invite.revoked", target_type="invite", target_id=invite.id,
                 email=invite.email)  # fmt: skip
    db.delete(invite)
    db.commit()


@router.patch("/members/{user_id}", response_model=UserOut)
def change_role(user_id: str, body: RoleIn, actor: AdminUser, db: DbSession) -> User:
    member = _member(db, actor, user_id)
    if "owner" in (body.role, member.role) and actor.role != "owner":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Only an owner can change owners.")
    if member.role == "owner" and body.role != "owner" and _owners(db, actor.workspace_id) <= 1:
        raise HTTPException(status.HTTP_409_CONFLICT, "A workspace needs at least one owner.")
    audit.record(db, actor, "member.role_changed", target_type="user", target_id=member.id,
                 email=member.email, old=member.role, new=body.role)  # fmt: skip
    member.role = body.role
    db.commit()
    return member


@router.delete("/members/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_member(user_id: str, actor: AdminUser, db: DbSession) -> None:
    member = _member(db, actor, user_id)
    if member.role == "owner" and actor.role != "owner":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Only an owner can remove an owner.")
    if member.role == "owner" and _owners(db, actor.workspace_id) <= 1:
        raise HTTPException(status.HTTP_409_CONFLICT, "A workspace needs at least one owner.")
    audit.record(db, actor, "member.removed", target_type="user", target_id=member.id,
                 email=member.email)  # fmt: skip
    db.delete(member)  # their conversations go too; datasets stay with the workspace
    db.commit()


@router.get("/audit", response_model=list[AuditOut])
def audit_log(
    actor: AdminUser,
    db: DbSession,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> list[AuditEvent]:
    stmt = (
        select(AuditEvent)
        .where(AuditEvent.workspace_id == actor.workspace_id)
        .order_by(AuditEvent.created_at.desc())
        .limit(limit)
    )
    return list(db.scalars(stmt))


@router.get("/activity", response_model=ActivityOut)
def workspace_activity(actor: CurrentUser, db: DbSession) -> ActivityOut:
    """Usage at a glance: questions asked, active people, busiest datasets."""
    ws = actor.workspace_id
    week_ago = datetime.now(UTC) - timedelta(days=7)
    q_base = (
        select(func.count())
        .select_from(Message)
        .join(Conversation, Conversation.id == Message.conversation_id)
        .where(Conversation.workspace_id == ws, Message.role == "user")
    )
    total = int(db.scalar(q_base) or 0)
    this_week = int(db.scalar(q_base.where(Message.created_at >= week_ago)) or 0)
    datasets = int(
        db.scalar(select(func.count()).select_from(Dataset).where(Dataset.workspace_id == ws)) or 0
    )
    conversations = int(
        db.scalar(
            select(func.count()).select_from(Conversation).where(Conversation.workspace_id == ws)
        )
        or 0
    )
    active = int(
        db.scalar(
            select(func.count(func.distinct(Conversation.user_id))).where(
                Conversation.workspace_id == ws
            )
        )
        or 0
    )
    top_ds = db.execute(
        select(Dataset.name, func.count(Message.id))
        .join(Conversation, Conversation.dataset_id == Dataset.id)
        .join(Message, Message.conversation_id == Conversation.id)
        .where(Dataset.workspace_id == ws, Message.role == "user")
        .group_by(Dataset.id)
        .order_by(func.count(Message.id).desc())
        .limit(5)
    ).all()
    people = db.execute(
        select(User.email, func.count(Message.id))
        .join(Conversation, Conversation.user_id == User.id)
        .join(Message, Message.conversation_id == Conversation.id)
        .where(Conversation.workspace_id == ws, Message.role == "user")
        .group_by(User.id)
        .order_by(func.count(Message.id).desc())
        .limit(5)
    ).all()
    return ActivityOut(
        questions_total=total,
        questions_this_week=this_week,
        datasets=datasets,
        conversations=conversations,
        active_members=active,
        top_datasets=[{"name": n, "questions": int(c)} for n, c in top_ds],
        busiest_people=[{"email": e, "questions": int(c)} for e, c in people],
    )


__all__ = ["CurrentUser", "hash_token", "router"]
