from datetime import UTC, datetime, timedelta

import jwt
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings
from app.core.security import (
    InvalidTokenError,
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from app.models import User
from tests.conftest import signup


def _me(client: TestClient, token: str) -> int:
    resp = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    return int(resp.status_code)


def test_password_hashing() -> None:
    h = hash_password("s3cret-pass")
    assert h != "s3cret-pass"
    assert verify_password("s3cret-pass", h)
    assert not verify_password("wrong", h)
    assert not verify_password("x" * 100, h)
    assert not verify_password("anything", None)


def test_token_round_trip(settings: Settings) -> None:
    token = create_access_token(settings, user_id="u1", workspace_id="w1")
    claims = decode_access_token(settings, token)
    assert (claims.user_id, claims.workspace_id) == ("u1", "w1")


def test_expired_token_rejected(client: TestClient, settings: Settings) -> None:
    acct = signup(client, "a@example.com")
    old = datetime.now(UTC) - timedelta(minutes=settings.access_token_expire_minutes + 1)
    token = create_access_token(
        settings, user_id=acct.user["id"], workspace_id=acct.workspace["id"], now=old
    )
    assert _me(client, token) == 401


def test_wrong_secret_rejected(client: TestClient, settings: Settings) -> None:
    acct = signup(client, "a@example.com")
    forged = settings.model_copy(update={"secret_key": SecretStr("attacker-" * 5)})
    token = create_access_token(forged, user_id=acct.user["id"], workspace_id=acct.workspace["id"])
    assert _me(client, token) == 401


def test_alg_none_rejected(client: TestClient) -> None:
    acct = signup(client, "a@example.com")
    now = datetime.now(UTC)
    payload = {
        "sub": acct.user["id"],
        "ws": acct.workspace["id"],
        "type": "access",
        "iat": now,
        "exp": now + timedelta(hours=1),
    }
    token = jwt.encode(payload, key="", algorithm="none")
    assert _me(client, token) == 401


def test_token_with_swapped_workspace_rejected(client: TestClient, settings: Settings) -> None:
    """A validly signed token that claims another workspace must not work."""
    a = signup(client, "a@example.com", workspace="A")
    b = signup(client, "b@example.com", workspace="B")
    token = create_access_token(settings, user_id=a.user["id"], workspace_id=b.workspace["id"])
    assert _me(client, token) == 401


def test_deleted_user_token_rejected(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    acct = signup(client, "a@example.com")
    with session_factory() as db:
        user = db.get(User, acct.user["id"])
        assert user is not None
        db.delete(user)
        db.commit()
    assert _me(client, acct.token) == 401


def test_garbage_token_raises(settings: Settings) -> None:
    with pytest.raises(InvalidTokenError):
        decode_access_token(settings, "not.a.jwt")
