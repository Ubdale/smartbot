"""Alembic environment: uses the app's DATABASE_URL and model metadata."""

from logging.config import fileConfig
from typing import Any, Literal

from alembic import context

from app.core.config import get_settings
from app.core.db import make_engine
from app.models import Base
from app.models.base import UTCDateTime

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata


def _render_item(type_: str, obj: Any, autogen_context: Any) -> str | Literal[False]:
    """Write app-specific column types as plain SQLAlchemy types in migrations."""
    if type_ == "type" and isinstance(obj, UTCDateTime):
        return "sa.DateTime(timezone=True)"
    return False


def _database_url() -> str:
    # Tests (or scripts) may override the URL explicitly via the Alembic config.
    override = config.get_main_option("sqlalchemy.url")
    return override or get_settings().database_url


def run_migrations_offline() -> None:
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
        render_item=_render_item,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = make_engine(_database_url())
    with engine.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,  # needed for ALTER TABLE on SQLite
            render_item=_render_item,
        )
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
