"""Browser auth: HttpOnly session cookie + CSRF header on writes."""

from fastapi.testclient import TestClient

from app.api.deps import AUTH_COOKIE, CSRF_HEADER


def _signup(client: TestClient) -> None:
    resp = client.post(
        "/auth/signup",
        json={"email": "web@acme.com", "password": "web-pass-123", "workspace_name": "Acme"},
    )
    assert resp.status_code == 201


def test_signup_and_login_set_httponly_cookie(client: TestClient) -> None:
    _signup(client)
    resp = client.post("/auth/login", data={"username": "web@acme.com", "password": "web-pass-123"})
    cookie = resp.headers["set-cookie"]
    assert cookie.startswith(f"{AUTH_COOKIE}=")
    assert "HttpOnly" in cookie
    assert "SameSite=lax" in cookie
    assert "Path=/" in cookie
    assert "Domain" not in cookie  # host-only: belongs to whichever domain served it


def test_cookie_authenticates_reads(client: TestClient) -> None:
    _signup(client)
    assert client.get("/auth/me").json()["email"] == "web@acme.com"
    assert client.get("/datasets").status_code == 200


def test_cookie_writes_need_csrf_header(client: TestClient) -> None:
    _signup(client)
    files = {"file": ("a.csv", b"x,y\n1,2\n")}
    blocked = client.post("/datasets", files=files)
    assert blocked.status_code == 403
    assert CSRF_HEADER in blocked.json()["detail"]
    allowed = client.post("/datasets", files=files, headers={CSRF_HEADER: "1"})
    assert allowed.status_code == 201


def test_bearer_requests_need_no_csrf_header(client: TestClient) -> None:
    resp = client.post(
        "/auth/signup",
        json={"email": "api@acme.com", "password": "api-pass-123", "workspace_name": "Acme"},
    )
    client.cookies.clear()
    token = resp.json()["access_token"]
    files = {"file": ("a.csv", b"x,y\n1,2\n")}
    resp = client.post("/datasets", files=files, headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 201


def test_logout_clears_cookie(client: TestClient) -> None:
    _signup(client)
    assert client.post("/auth/logout").status_code == 204
    assert client.get("/auth/me").status_code == 401


def test_prod_cookie_is_secure(client: TestClient, app: object, settings: object) -> None:
    from fastapi import FastAPI

    from app.core.config import Settings, get_settings

    assert isinstance(app, FastAPI)
    assert isinstance(settings, Settings)
    prod = settings.model_copy(update={"app_env": "prod"})
    app.dependency_overrides[get_settings] = lambda: prod
    resp = client.post(
        "/auth/signup",
        json={"email": "p@acme.com", "password": "prod-pass-123", "workspace_name": "P"},
    )
    assert "Secure" in resp.headers["set-cookie"]
