from typing import TYPE_CHECKING

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db import Base
from app.models.base import IdMixin, TimestampMixin

if TYPE_CHECKING:
    from app.models.user import User


class Workspace(IdMixin, TimestampMixin, Base):
    """One workspace per company. Every piece of customer data belongs to exactly one."""

    __tablename__ = "workspaces"

    name: Mapped[str] = mapped_column(String(120))

    users: Mapped[list["User"]] = relationship(back_populates="workspace")
