"""SQLAlchemy models. Import every model module here so Alembic autogenerate sees it."""

from app.core.db import Base
from app.models.chat import Conversation, Message, UsageEvent
from app.models.dataset import Dataset
from app.models.learning import GlossaryTerm, LearnedExample
from app.models.team import AuditEvent, Invite
from app.models.user import User
from app.models.workspace import Workspace

__all__ = [
    "AuditEvent",
    "Base",
    "Conversation",
    "Dataset",
    "GlossaryTerm",
    "Invite",
    "LearnedExample",
    "Message",
    "UsageEvent",
    "User",
    "Workspace",
]
