from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import inspect

from app.core.db import make_engine

ROOT = Path(__file__).resolve().parent.parent


def test_upgrade_and_downgrade(tmp_path: Path) -> None:
    url = f"sqlite:///{(tmp_path / 'mig.db').as_posix()}"
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", url)

    command.upgrade(cfg, "head")
    engine = make_engine(url)
    assert "alembic_version" in inspect(engine).get_table_names()

    command.downgrade(cfg, "base")
    engine.dispose()
