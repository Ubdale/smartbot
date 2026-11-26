"""Dashboards: a named set of saved questions, re-run on the latest data when viewed."""

from typing import Any

from sqlalchemy import JSON, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base
from app.models.base import IdMixin, TimestampMixin


class Dashboard(IdMixin, TimestampMixin, Base):
    __tablename__ = "dashboards"

    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"), index=True
    )
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    # [{"dataset_id", "question", "plan"}]: each tile is re-run on the current data.
    tiles: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
