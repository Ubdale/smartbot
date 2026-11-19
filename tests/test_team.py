"""Teams: invite links, roles, removal, password changes, sessions, audit log."""

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import update
from sqlalchemy.orm import Session, sessionmaker

from app.models import Invite
from tests.conftest import Account, signup


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def invite(client: TestClient, by: Account, email: str, role: str = "member") -> dict[str, Any]:
    resp = client.post(
        "/workspaces/me/invites", headers=by.headers, json={"email": email, "role": role}
    )
    assert resp.status_code == 201, resp.text
    return dict(resp.json())


def accept(client: TestClient, token: str, password: str = "joining-pass-1") -> str:
    resp = client.post("/auth/invites/accept", json={"token": token, "password": password})
    assert resp.status_code == 200, resp.text
    return str(resp.json()["access_token"])


@pytest.fixture
def owner(client: TestClient) -> Account:
    return signup(client, "owner@acme.com", workspace="Acme")


def test_invite_link_flow(client: TestClient, owner: Account) -> None:
    created = invite(client, owner, "Sara@Acme.com", role="admin")
    info = client.get(f"/auth/invites/{created['token']}").json()
    assert info == {"email": "sara@acme.com", "role": "admin", "workspace_name": "Acme"}

    token = accept(client, created["token"])
    me = client.get("/auth/me", headers=bearer(token)).json()
    assert (me["email"], me["role"], me["workspace_id"]) == (
        "sara@acme.com", "admin", owner.workspace["id"]
    )  # fmt: skip
    members = client.get("/workspaces/me/members", headers=owner.headers).json()
    assert {m["email"] for m in members} == {"owner@acme.com", "sara@acme.com"}
    # links work once
    again = client.post(
        "/auth/invites/accept", json={"token": created["token"], "password": "x" * 9}
    )
    assert again.status_code == 404


def test_expired_and_unknown_invites(
    client: TestClient, owner: Account, session_factory: sessionmaker[Session]
) -> None:
    created = invite(client, owner, "late@acme.com")
    with session_factory() as db:
        db.execute(update(Invite).values(expires_at=datetime.now(UTC) - timedelta(minutes=1)))
        db.commit()
    assert client.get(f"/auth/invites/{created['token']}").status_code == 404
    assert client.get("/auth/invites/not-a-real-token-123").status_code == 404


def test_cannot_invite_an_existing_account(client: TestClient, owner: Account) -> None:
    signup(client, "taken@other.com", workspace="Other")
    resp = client.post(
        "/workspaces/me/invites", headers=owner.headers, json={"email": "taken@other.com"}
    )
    assert resp.status_code == 409


def test_members_cannot_manage_the_team(client: TestClient, owner: Account) -> None:
    member = bearer(accept(client, invite(client, owner, "m@acme.com")["token"]))
    assert (
        client.post(
            "/workspaces/me/invites", headers=member, json={"email": "x@acme.com"}
        ).status_code
        == 403
    )
    assert client.get("/workspaces/me/audit", headers=member).status_code == 403
    assert client.get("/workspaces/me/invites", headers=member).status_code == 403
    # ...but can use the data
    assert client.get("/datasets", headers=member).status_code == 200


def test_roles_and_the_last_owner(client: TestClient, owner: Account) -> None:
    admin = bearer(accept(client, invite(client, owner, "a@acme.com", "admin")["token"]))
    member_token = accept(client, invite(client, owner, "m@acme.com")["token"])
    member_id = client.get("/auth/me", headers=bearer(member_token)).json()["id"]
    owner_id = owner.user["id"]

    # admins can't touch owners or create them
    assert client.patch(f"/workspaces/me/members/{member_id}", headers=admin,
                        json={"role": "owner"}).status_code == 403  # fmt: skip
    assert client.delete(f"/workspaces/me/members/{owner_id}", headers=admin).status_code == 403
    # the only owner can't step down or leave
    assert client.patch(f"/workspaces/me/members/{owner_id}", headers=owner.headers,
                        json={"role": "member"}).status_code == 409  # fmt: skip
    # owners can promote; then the first owner may step down
    promoted = client.patch(f"/workspaces/me/members/{member_id}", headers=owner.headers,
                            json={"role": "owner"})  # fmt: skip
    assert promoted.json()["role"] == "owner"
    assert client.patch(f"/workspaces/me/members/{owner_id}", headers=owner.headers,
                        json={"role": "admin"}).status_code == 200  # fmt: skip


def test_removed_member_is_signed_out(client: TestClient, owner: Account) -> None:
    token = accept(client, invite(client, owner, "bye@acme.com")["token"])
    user_id = client.get("/auth/me", headers=bearer(token)).json()["id"]
    assert (
        client.delete(f"/workspaces/me/members/{user_id}", headers=owner.headers).status_code == 204
    )
    assert client.get("/auth/me", headers=bearer(token)).status_code == 401


def test_password_change_signs_out_other_sessions(client: TestClient, owner: Account) -> None:
    old_session = owner.headers
    resp = client.post(
        "/auth/change-password",
        headers=old_session,
        json={"current_password": owner.password, "new_password": "brand-new-pass-1"},
    )
    assert resp.status_code == 200
    assert client.get("/auth/me", headers=old_session).status_code == 401  # stolen token: dead
    assert client.get("/auth/me", headers=bearer(resp.json()["access_token"])).status_code == 200
    wrong = client.post(
        "/auth/change-password",
        headers=bearer(resp.json()["access_token"]),
        json={"current_password": "nope-nope-1", "new_password": "another-pass-1"},
    )
    assert wrong.status_code == 400
    login = client.post(
        "/auth/login", data={"username": "owner@acme.com", "password": "brand-new-pass-1"}
    )
    assert login.status_code == 200


def test_logout_everywhere(client: TestClient, owner: Account) -> None:
    assert client.post("/auth/logout-everywhere", headers=owner.headers).status_code == 204
    assert client.get("/auth/me", headers=owner.headers).status_code == 401


def test_audit_log_records_and_is_isolated(client: TestClient, owner: Account) -> None:
    invite(client, owner, "audit@acme.com")
    client.post("/datasets", headers=owner.headers, files={"file": ("a.csv", b"x,y\n1,2\n")})
    events = client.get("/workspaces/me/audit", headers=owner.headers).json()
    actions = [e["action"] for e in events]
    assert {"workspace.created", "invite.created", "dataset.uploaded"} <= set(actions)
    upload = next(e for e in events if e["action"] == "dataset.uploaded")
    assert upload["actor_email"] == "owner@acme.com"
    assert upload["details"]["file"] == "a.csv"

    other = signup(client, "boss@other.com", workspace="Other")
    other_actions = {
        e["action"] for e in client.get("/workspaces/me/audit", headers=other.headers).json()
    }
    assert other_actions == {"workspace.created"}


def test_invites_and_members_are_workspace_scoped(client: TestClient, owner: Account) -> None:
    created = invite(client, owner, "x@acme.com")
    other = signup(client, "o@other.com", workspace="Other")
    assert (
        client.delete(f"/workspaces/me/invites/{created['id']}", headers=other.headers).status_code
        == 404
    )
    assert (
        client.delete(
            f"/workspaces/me/members/{owner.user['id']}", headers=other.headers
        ).status_code
        == 404
    )
