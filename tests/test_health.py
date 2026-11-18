from collections.abc import Iterator

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from app.core.db import get_db


def test_health_ok(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["database"] == "ok"
    assert body["env"] == "test"
    assert body["llm_provider"] == "ollama"


def test_health_reports_db_failure(app: FastAPI) -> None:
    class BrokenSession:
        def execute(self, *_args: object, **_kwargs: object) -> None:
            raise OperationalError("SELECT 1", {}, Exception("db down"))

    def _broken_db() -> Iterator[BrokenSession]:
        yield BrokenSession()

    app.dependency_overrides[get_db] = _broken_db
    with TestClient(app) as c:
        resp = c.get("/health")
    assert resp.status_code == 503
    assert resp.json()["status"] == "degraded"
    assert resp.json()["database"] == "error"
