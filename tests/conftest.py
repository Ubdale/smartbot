import os

# Before any app import: real env vars beat `.env`, so the module-level engine in
# app.core.db can never point at the developer's real database (e.g. Neon) during tests.
os.environ["DATABASE_URL"] = "sqlite://"
os.environ["APP_ENV"] = "test"

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings, get_settings
from app.core.db import get_db, make_engine
from app.main import create_app
from app.models import Base

# Set TEST_DATABASE_URL to run the suite against a throwaway Postgres (CI does this).
# Its tables are dropped and recreated for every test, so never point it at real data.
TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        app_env="test",
        database_url=TEST_DATABASE_URL or f"sqlite:///{(tmp_path / 'test.db').as_posix()}",
        data_dir=tmp_path / "data",
        secret_key="test-secret-key-that-is-long-enough-for-hs256",
    )


@pytest.fixture
def engine(settings: Settings) -> Iterator[Engine]:
    eng = make_engine(settings.database_url)
    Base.metadata.drop_all(eng)
    Base.metadata.create_all(eng)
    yield eng
    Base.metadata.drop_all(eng)
    eng.dispose()


@pytest.fixture
def session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


@pytest.fixture
def app(settings: Settings, session_factory: sessionmaker[Session]) -> FastAPI:
    def _get_db() -> Iterator[Session]:
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    application = create_app()
    application.dependency_overrides[get_settings] = lambda: settings
    application.dependency_overrides[get_db] = _get_db
    return application


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c


@dataclass
class Account:
    email: str
    password: str
    token: str
    user: dict[str, Any]
    workspace: dict[str, Any]

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}


def signup(
    client: TestClient,
    email: str,
    workspace: str = "Acme",
    password: str = "correct-horse-battery",
) -> Account:
    resp = client.post(
        "/auth/signup",
        json={"email": email, "password": password, "workspace_name": workspace},
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    return Account(email, password, body["access_token"], body["user"], body["workspace"])
