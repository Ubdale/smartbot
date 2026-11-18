from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import inspect

from app.core.db import make_engine

ROOT = Path(__file__).resolve().parent.parent


def _config(tmp_path: Path) -> tuple[Config, str]:
    url = f"sqlite:///{(tmp_path / 'mig.db').as_posix()}"
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", url)
    return cfg, url


def test_upgrade_and_downgrade(tmp_path: Path) -> None:
    cfg, url = _config(tmp_path)
    command.upgrade(cfg, "head")
    engine = make_engine(url)
    assert {"alembic_version", "users", "workspaces"} <= set(inspect(engine).get_table_names())

    command.downgrade(cfg, "base")
    assert set(inspect(engine).get_table_names()) == {"alembic_version"}
    engine.dispose()


def test_models_match_migrations(tmp_path: Path) -> None:
    """Fails if someone changes a model without generating a migration."""
    cfg, _ = _config(tmp_path)
    command.upgrade(cfg, "head")
    command.check(cfg)  # raises if autogenerate would produce any operations
