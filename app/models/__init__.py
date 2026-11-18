"""SQLAlchemy models. Import every model module here so Alembic autogenerate sees it."""

from app.core.db import Base

__all__ = ["Base"]
