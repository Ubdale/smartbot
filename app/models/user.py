from typing import TYPE_CHECKING, Literal

from sqlalchemy import ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db import Base
from app.models.base import IdMixin, TimestampMixin

if TYPE_CHECKING:
    from app.models.workspace import Workspace

Role = Literal["owner", "admin", "member"]


class User(IdMixin, TimestampMixin, Base):
    __tablename__ = "users"

    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(100))
    role: Mapped[str] = mapped_column(String(20), default="member")
    # Bumped on password change / "log out everywhere": older tokens stop working.
    token_version: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"), index=True
    )

    workspace: Mapped["Workspace"] = relationship(back_populates="users")
