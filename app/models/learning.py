"""What AskData learns from its users: worked examples and team vocabulary, per file."""

from typing import Any

from sqlalchemy import JSON, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base
from app.models.base import IdMixin, TimestampMixin


class LearnedExample(IdMixin, TimestampMixin, Base):
    """A question whose plan a user marked as right (thumbs up). Shown to the model as an
    example when someone later asks something similar about the same file."""

    __tablename__ = "learned_examples"

    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"), index=True
    )
    dataset_id: Mapped[str] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), index=True
    )
    question: Mapped[str] = mapped_column(Text)
    question_key: Mapped[str] = mapped_column(String(500), index=True)
    plan: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_by_id: Mapped[str | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )


class GlossaryTerm(IdMixin, TimestampMixin, Base):
    """A word the team uses for something in a file: "sales" means revenue."""

    __tablename__ = "glossary_terms"

    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"), index=True
    )
    dataset_id: Mapped[str] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), index=True
    )
    term: Mapped[str] = mapped_column(String(100))
    meaning: Mapped[str] = mapped_column(String(300))
    created_by_id: Mapped[str | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
