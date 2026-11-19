import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.core.ratelimit import RateLimiter
from tests.conftest import signup


def test_limiter_window() -> None:
    limiter = RateLimiter()
    for _ in range(3):
        limiter.check("u1", 3)
    with pytest.raises(HTTPException) as exc:
        limiter.check("u1", 3)
    assert exc.value.status_code == 429
    limiter.check("u2", 3)  # per user
    limiter.check("u1", 0)  # 0 = unlimited


def test_ask_is_limited_per_user(app: FastAPI, client: TestClient, settings: Settings) -> None:
    app.dependency_overrides[get_settings] = lambda: settings.model_copy(
        update={"questions_per_minute": 2}
    )
    acct = signup(client, "fast@acme.com")
    body = {"question": "hi", "dataset_id": "missing"}
    codes = [
        client.post("/chat/ask", headers=acct.headers, json=body).status_code for _ in range(3)
    ]
    assert codes == [404, 404, 429]
    resp = client.post("/chat/ask", headers=acct.headers, json=body)
    assert "wait" in resp.json()["detail"]
    assert int(resp.headers["retry-after"]) > 0
