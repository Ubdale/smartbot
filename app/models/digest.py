"""Scheduled reports: questions re-run on a schedule and sent as a digest."""

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base
from app.models.base import IdMixin, TimestampMixin


class Digest(IdMixin, TimestampMixin, Base):
    """A saved report: questions about one dataset, re-run daily or weekly."""

    __tablename__ = "digests"

    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"), index=True
    )
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    dataset_id: Mapped[str] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), index=True
    )
    title: Mapped[str] = mapped_column(String(200))
    # [{"question": str, "plan": {...}}]: re-run each time, so the numbers are current.
    items: Mapped[list[dict[str, Any]]] = mapped_column(JSON)
    cadence: Mapped[str] = mapped_column(String(10))  # daily | weekly
    weekday: Mapped[int] = mapped_column(Integer, default=0)  # 0 = Monday (weekly only)
    hour: Mapped[int] = mapped_column(Integer, default=9)  # local time, 0-23
    minute: Mapped[int] = mapped_column(Integer, default=0, server_default="0")  # 0-59
    timezone: Mapped[str] = mapped_column(String(64), default="Asia/Karachi")
    recipients: Mapped[list[str]] = mapped_column(JSON)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    next_run_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    last_run_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class DigestRun(IdMixin, TimestampMixin, Base):
    """One sending of a digest: what it said, and whether the email went out."""

    __tablename__ = "digest_runs"

    digest_id: Mapped[str] = mapped_column(ForeignKey("digests.id", ondelete="CASCADE"), index=True)
    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"), index=True
    )
    dataset_version: Mapped[int] = mapped_column(Integer)
    # [{"question", "summary", "columns", "rows", "truncated", "headline", "previous"}]
    content: Mapped[list[dict[str, Any]]] = mapped_column(JSON)
    changes: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    emailed: Mapped[bool] = mapped_column(Boolean, default=False)
    # Why the email didn't go out (no email service set up, a delivery error, ...).
    email_note: Mapped[str | None] = mapped_column(Text, nullable=True)
