"""Dashboards: pin answers and view them re-run on current data."""

from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.deps import get_today
from app.llm.factory import get_llm
from app.query.executor import clear_cache
from tests.conftest import Account, signup
from tests.fakes import FakeLLM

SAMPLE = Path(__file__).resolve().parent.parent / "sample_data"
TODAY = date(2026, 11, 19)
BY_REGION = {"metric": "sum", "metric_column": "revenue", "group_by": ["region"], "chart": "bar"}


@pytest.fixture(autouse=True)
def _setup(app: FastAPI) -> Iterator[None]:
    clear_cache()
    app.dependency_overrides[get_today] = lambda: TODAY
    yield
    clear_cache()


def test_pin_and_view_dashboard(app: FastAPI, client: TestClient) -> None:
    acct: Account = signup(client, "owner@acme.com", workspace="Acme")
    ds = client.post(
        "/datasets",
        headers=acct.headers,
        files={"file": ("sales.csv", (SAMPLE / "sales_orders.csv").read_bytes())},
    ).json()["id"]
    app.dependency_overrides[get_llm] = lambda: FakeLLM(
        [BY_REGION, {"metric": "count"}], summary="North leads."
    )

    def ask(q: str, **ids: str) -> dict[str, Any]:
        r = client.post("/chat/ask", headers=acct.headers, json={"question": q, **ids})
        assert r.status_code == 200, r.text
        return r.json()

    a1 = ask("revenue by region", dataset_id=ds)
    conv = a1["conversation_id"]
    a2 = ask("how many orders", conversation_id=conv)

    # Pin the first answer to a new dashboard.
    pin = client.post(
        "/dashboards/pin",
        headers=acct.headers,
        json={"message_id": a1["reply"]["id"], "name": "Sales overview"},
    ).json()
    assert pin["name"] == "Sales overview" and pin["tiles"] == 1
    board = pin["id"]
    # Pin the second to the same dashboard.
    client.post(
        "/dashboards/pin",
        headers=acct.headers,
        json={"message_id": a2["reply"]["id"], "dashboard_id": board},
    )

    view = client.get(f"/dashboards/{board}", headers=acct.headers).json()
    assert view["name"] == "Sales overview" and len(view["tiles"]) == 2
    first = view["tiles"][0]
    assert first["question"] == "revenue by region"
    assert first["columns"] == ["region", "sum_revenue"] and first["chart"]["type"] == "bar"
    assert len(first["rows"]) == 4  # re-run on the live data

    # List, remove a tile, delete.
    assert client.get("/dashboards", headers=acct.headers).json()[0]["tiles"] == 2
    assert client.delete(f"/dashboards/{board}/tiles/1", headers=acct.headers).json()["tiles"] == 1

    other = signup(client, "spy@other.com", workspace="Other")
    assert client.get(f"/dashboards/{board}", headers=other.headers).status_code == 404
    assert client.delete(f"/dashboards/{board}", headers=acct.headers).status_code == 204


def test_cannot_pin_a_file_overview(app: FastAPI, client: TestClient) -> None:
    acct: Account = signup(client, "owner@acme.com", workspace="Acme")
    ds = client.post(
        "/datasets",
        headers=acct.headers,
        files={"file": ("sales.csv", (SAMPLE / "sales_orders.csv").read_bytes())},
    ).json()["id"]
    app.dependency_overrides[get_llm] = lambda: FakeLLM([{"metric": "describe"}], summary="")
    reply = client.post(
        "/chat/ask",
        headers=acct.headers,
        json={"question": "what is this file about?", "dataset_id": ds},
    ).json()["reply"]
    resp = client.post("/dashboards/pin", headers=acct.headers, json={"message_id": reply["id"]})
    assert resp.status_code == 400
