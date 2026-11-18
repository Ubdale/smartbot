import pytest
from fastapi.testclient import TestClient

from tests.conftest import signup


def test_signup_creates_owner_and_workspace(client: TestClient) -> None:
    acct = signup(client, "Alice@Example.com", workspace="  Acme Ltd  ")
    assert acct.user["email"] == "alice@example.com"  # normalised
    assert acct.user["role"] == "owner"
    assert acct.workspace["name"] == "Acme Ltd"
    assert acct.user["workspace_id"] == acct.workspace["id"]
    assert "password_hash" not in acct.user


def test_signup_duplicate_email_is_conflict(client: TestClient) -> None:
    signup(client, "alice@example.com")
    resp = client.post(
        "/auth/signup",
        json={"email": "ALICE@example.com", "password": "another-pass-1", "workspace_name": "X"},
    )
    assert resp.status_code == 409


@pytest.mark.parametrize(
    "payload",
    [
        {"email": "not-an-email", "password": "long-enough-1", "workspace_name": "A"},
        {"email": "a@example.com", "password": "short", "workspace_name": "A"},
        {"email": "a@example.com", "password": "x" * 73, "workspace_name": "A"},
        {"email": "a@example.com", "password": "long-enough-1", "workspace_name": "   "},
    ],
)
def test_signup_validation(client: TestClient, payload: dict[str, str]) -> None:
    assert client.post("/auth/signup", json=payload).status_code == 422


def test_login_and_me(client: TestClient) -> None:
    acct = signup(client, "bob@example.com")
    resp = client.post(
        "/auth/login", data={"username": " BOB@example.com ", "password": acct.password}
    )
    assert resp.status_code == 200
    token = resp.json()["access_token"]
    assert resp.json()["token_type"] == "bearer"

    me = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["id"] == acct.user["id"]


@pytest.mark.parametrize(
    ("username", "password"),
    [("bob@example.com", "wrong-password"), ("nobody@example.com", "correct-horse-battery")],
)
def test_login_failures_look_identical(client: TestClient, username: str, password: str) -> None:
    signup(client, "bob@example.com")
    resp = client.post("/auth/login", data={"username": username, "password": password})
    assert resp.status_code == 401
    assert resp.json()["detail"] == "Incorrect email or password"


def test_me_requires_token(client: TestClient) -> None:
    resp = client.get("/auth/me")
    assert resp.status_code == 401
    assert resp.headers["www-authenticate"] == "Bearer"


def test_timestamps_are_utc(client: TestClient) -> None:
    acct = signup(client, "tz@example.com")
    assert acct.user["created_at"].endswith("Z") or acct.user["created_at"].endswith("+00:00")
