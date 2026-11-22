"""Saved questions: a starred answer you can re-run on the latest data anytime."""

from typing import Any

from sqlalchemy import JSON, ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base
from app.models.base import IdMixin, TimestampMixin


class SavedAnswer(IdMixin, TimestampMixin, Base):
    """A question the user starred. Stores the question and its plan, so it re-runs on
    the current data (not a frozen copy)."""

    __tablename__ = "saved_answers"

    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"), index=True
    )
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    dataset_id: Mapped[str] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), index=True
    )
    question: Mapped[str] = mapped_column(Text)
    plan: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
