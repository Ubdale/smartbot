"""Workspace isolation: a user can never see another workspace or its members."""

import pytest
from fastapi.testclient import TestClient

from tests.conftest import Account, signup


@pytest.fixture
def a(client: TestClient) -> Account:
    return signup(client, "alice@a-corp.com", workspace="A Corp")


@pytest.fixture
def b(client: TestClient) -> Account:
    return signup(client, "bob@b-corp.com", workspace="B Corp")


def test_each_user_sees_only_own_workspace(client: TestClient, a: Account, b: Account) -> None:
    assert client.get("/workspaces/me", headers=a.headers).json()["id"] == a.workspace["id"]
    assert client.get("/workspaces/me", headers=b.headers).json()["id"] == b.workspace["id"]


def test_can_read_own_workspace_by_id(client: TestClient, a: Account) -> None:
    resp = client.get(f"/workspaces/{a.workspace['id']}", headers=a.headers)
    assert resp.status_code == 200


def test_user_a_cannot_read_workspace_b(client: TestClient, a: Account, b: Account) -> None:
    resp = client.get(f"/workspaces/{b.workspace['id']}", headers=a.headers)
    # 404 (not 403) so A can't even learn that B's workspace exists.
    assert resp.status_code == 404
    unknown = client.get("/workspaces/00000000-0000-0000-0000-000000000000", headers=a.headers)
    assert resp.json() == unknown.json()


def test_members_list_is_scoped(client: TestClient, a: Account, b: Account) -> None:
    a_members = client.get("/workspaces/me/members", headers=a.headers).json()
    b_members = client.get("/workspaces/me/members", headers=b.headers).json()
    assert [m["email"] for m in a_members] == ["alice@a-corp.com"]
    assert [m["email"] for m in b_members] == ["bob@b-corp.com"]


@pytest.mark.parametrize("path", ["/workspaces/me", "/workspaces/me/members", "/workspaces/x"])
def test_workspace_routes_require_auth(client: TestClient, path: str) -> None:
    assert client.get(path).status_code == 401
