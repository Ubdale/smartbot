"""Record who did what in a workspace. Never stores secrets or file contents."""

from typing import Any

from sqlalchemy.orm import Session

from app.models import AuditEvent, User


def record(
    db: Session,
    actor: User | None,
    action: str,
    *,
    workspace_id: str | None = None,
    target_type: str | None = None,
    target_id: str | None = None,
    **details: Any,
) -> None:
    """Add an audit event to the session; it's saved with the caller's commit."""
    ws = workspace_id or (actor.workspace_id if actor else None)
    if ws is None:
        return
    db.add(
        AuditEvent(
            workspace_id=ws,
            user_id=actor.id if actor else None,
            actor_email=actor.email if actor else None,
            action=action,
            target_type=target_type,
            target_id=target_id,
            details=details or None,
        )
    )
