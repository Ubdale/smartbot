from typing import Any

from sqlalchemy import JSON, BigInteger, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base
from app.models.base import IdMixin, TimestampMixin


class Dataset(IdMixin, TimestampMixin, Base):
    """An uploaded table, stored as Parquet under its workspace's folder."""

    __tablename__ = "datasets"

    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"), index=True
    )
    created_by_id: Mapped[str | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    name: Mapped[str] = mapped_column(String(200))
    original_filename: Mapped[str] = mapped_column(String(255))
    file_format: Mapped[str] = mapped_column(String(10))  # csv | xlsx
    sheet_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    row_count: Mapped[int] = mapped_column(BigInteger)
    # List of ColumnProfile dicts: name, type, kind, null_count, distinct_count, samples, min, max
    columns: Mapped[list[dict[str, Any]]] = mapped_column(JSON)
    # Bumped whenever the data changes; part of the query-cache key (Milestone 4).
    version: Mapped[int] = mapped_column(Integer, default=1)
    # "table" (questions computed with SQL) or "document" (answers quoted from text).
    kind: Mapped[str] = mapped_column(String(20), default="table", server_default="table")
    # How the table was obtained when it wasn't read directly, e.g. "ai_vision" for a
    # table an AI model read from an image. Shown to users as "please check".
    extraction: Mapped[str | None] = mapped_column(String(20), nullable=True)
