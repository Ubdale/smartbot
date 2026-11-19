"""Scheduled digests: schedules, re-running saved questions, email, and the timer."""

import json
from collections.abc import Iterator
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy.orm import Session, sessionmaker

from app.api.deps import get_today
from app.core.config import Settings, get_settings
from app.digests import is_schedule_request, next_run, parse_schedule, send_digest_email
from app.llm.factory import get_llm
from app.models import Digest, DigestRun
from app.query.executor import clear_cache
from tests.conftest import Account, signup
from tests.fakes import FakeLLM

SAMPLE = Path(__file__).resolve().parent.parent / "sample_data"
TODAY = date(2026, 11, 19)
REVENUE = {"metric": "sum", "metric_column": "revenue"}
BY_REGION = {"metric": "sum", "metric_column": "revenue", "group_by": ["region"], "chart": "bar"}


def test_next_run_in_the_users_timezone() -> None:
    # Thursday 2026-11-19 10:00 UTC = 15:00 in Karachi (UTC+5).
    now = datetime(2026, 11, 19, 10, 0, tzinfo=UTC)
    weekly = next_run("weekly", 0, 9, "Asia/Karachi", now)  # Monday 09:00 Karachi
    assert weekly == datetime(2026, 11, 23, 4, 0, tzinfo=UTC)
    daily = next_run("daily", 0, 9, "Asia/Karachi", now)  # tomorrow 09:00 Karachi
    assert daily == datetime(2026, 11, 20, 4, 0, tzinfo=UTC)
    later_today = next_run("daily", 0, 18, "Asia/Karachi", now)
    assert later_today == datetime(2026, 11, 19, 13, 0, tzinfo=UTC)
    assert next_run("weekly", 0, 9, "Not/AZone", now).tzinfo is not None  # falls back to UTC


def test_schedule_requests() -> None:
    assert parse_schedule("send this every Friday at 5pm").__dict__ == {
        "cadence": "weekly",
        "weekday": 4,
        "hour": 17,
    }
    assert parse_schedule("email me daily").cadence == "daily"
    assert parse_schedule("top 10 customers every tuesday").hour == 9  # "10" isn't a time
    for q in ["send me this every Monday at 9", "email this weekly", "har monday bhejo"]:
        assert is_schedule_request(q), q
    for q in ["weekly revenue trend", "orders every monday", "send me top 10 customers"]:
        assert not is_schedule_request(q), q


@pytest.fixture(autouse=True)
def _setup(app: FastAPI) -> Iterator[None]:
    clear_cache()
    app.dependency_overrides[get_today] = lambda: TODAY
    yield
    clear_cache()


def _with_settings(app: FastAPI, settings: Settings, **changes: Any) -> Settings:
    updated = settings.model_copy(update=changes)
    app.dependency_overrides[get_settings] = lambda: updated
    return updated


def test_schedule_from_chat_run_now_and_inbox(
    app: FastAPI, client: TestClient, settings: Settings
) -> None:
    acct: Account = signup(client, "owner@acme.com", workspace="Acme")
    ds = client.post(
        "/datasets",
        headers=acct.headers,
        files={"file": ("sales.csv", (SAMPLE / "sales_orders.csv").read_bytes())},
    ).json()["id"]
    app.dependency_overrides[get_llm] = lambda: FakeLLM([BY_REGION], summary="North leads.")
    first = client.post(
        "/chat/ask", headers=acct.headers, json={"question": "Revenue by region", "dataset_id": ds}
    ).json()
    conv = first["conversation_id"]

    reply = client.post(
        "/chat/ask",
        headers=acct.headers,
        json={"question": "send me this every Monday at 9am", "conversation_id": conv},
    ).json()["reply"]
    assert "every Monday at 09:00 (Asia/Karachi)" in reply["content"]
    assert "Email isn't set up yet" in reply["content"]

    digests = client.get("/digests", headers=acct.headers).json()
    assert len(digests) == 1
    d = digests[0]
    assert d["questions"] == ["Revenue by region"] and d["recipients"] == ["owner@acme.com"]

    run = client.post(f"/digests/{d['id']}/run", headers=acct.headers)
    assert run.status_code == 200, run.text
    body = run.json()
    item = body["content"][0]
    assert item["columns"] == ["region", "sum_revenue"] and len(item["rows"]) == 4
    assert body["emailed"] is False and "RESEND_API_KEY" in body["email_note"]
    assert client.get("/digests/runs", headers=acct.headers).json()[0]["id"] == body["id"]

    # Edit, then a teammate/outsider can't see or touch it.
    patched = client.patch(
        f"/digests/{d['id']}", headers=acct.headers, json={"cadence": "daily", "hour": 18}
    ).json()
    assert patched["schedule"] == "every day at 18:00 (Asia/Karachi)"
    other = signup(client, "spy@other.com", workspace="Other")
    assert client.delete(f"/digests/{d['id']}", headers=other.headers).status_code == 404
    assert client.get("/digests", headers=other.headers).json() == []


def test_timer_sends_due_digests_and_compares_with_last_time(
    app: FastAPI,
    client: TestClient,
    settings: Settings,
    session_factory: sessionmaker[Session],
) -> None:
    acct: Account = signup(client, "owner@acme.com", workspace="Acme")
    ds = client.post(
        "/datasets",
        headers=acct.headers,
        files={"file": ("sales.csv", (SAMPLE / "sales_orders.csv").read_bytes())},
    ).json()["id"]
    app.dependency_overrides[get_llm] = lambda: FakeLLM([REVENUE], summary="Total.")
    reply = client.post(
        "/chat/ask", headers=acct.headers, json={"question": "Total revenue", "dataset_id": ds}
    ).json()["reply"]
    d = client.post("/digests", headers=acct.headers, json={"message_id": reply["id"]}).json()

    # No secret configured: the endpoint doesn't exist.
    assert client.post("/internal/digests/tick").status_code == 404
    _with_settings(app, settings, digest_cron_secret=SecretStr("s3cret"))
    assert (
        client.post("/internal/digests/tick", headers={"X-Cron-Secret": "nope"}).status_code == 401
    )

    with session_factory() as db:  # make it due
        digest = db.get(Digest, d["id"])
        assert digest is not None
        digest.next_run_at = datetime(2026, 1, 1, tzinfo=UTC)
        db.commit()
    for _ in range(2):
        resp = client.post("/internal/digests/tick", headers={"X-Cron-Secret": "s3cret"})
        assert resp.json() == {"sent": 1, "emailed": 0}
        with session_factory() as db:
            digest = db.get(Digest, d["id"])
            assert digest is not None
            assert digest.next_run_at.replace(tzinfo=UTC) > datetime.now(UTC)
            digest.next_run_at = datetime(2026, 1, 1, tzinfo=UTC)
            db.commit()
    with session_factory() as db:
        runs = db.query(DigestRun).order_by(DigestRun.created_at).all()
        assert len(runs) == 2
        second = runs[1].content[0]
        assert second["headline"] == second["previous"]  # compared with the run before


def test_email_goes_out_through_resend(settings: Settings) -> None:
    from app.models import Dataset, User

    sent: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://api.resend.com/emails"
        assert request.headers["authorization"] == "Bearer re_test"
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"id": "x"})

    s = settings.model_copy(
        update={"resend_api_key": SecretStr("re_test"), "app_url": "https://app.test"}
    )
    digest = Digest(
        title="Weekly revenue",
        recipients=["a@b.com"],
        cadence="weekly",
        weekday=0,
        hour=9,
        timezone="Asia/Karachi",
        items=[],
    )
    run = DigestRun(
        dataset_version=1,
        created_at=datetime(2026, 11, 23, tzinfo=UTC),
        content=[
            {
                "question": "Total revenue",
                "summary": "Total revenue is 1,200.",
                "columns": ["sum"],
                "rows": [[1200.0]],
                "headline": 1200.0,
                "previous": 1000.0,
            },
            {
                "question": "By region",
                "summary": "North leads.",
                "columns": ["region", "sum"],
                "rows": [["North", 800.0], ["South", 400.0]],
                "headline": None,
                "previous": None,
            },
        ],
        changes={"added": 3, "removed": 0, "changed": 1},
    )
    ds = Dataset(name="Sales")
    ok, note = send_digest_email(
        s,
        digest,
        run,
        ds,
        User(email="a@b.com"),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert ok and note is None
    mail = sent[0]
    assert mail["to"] == ["a@b.com"] and mail["subject"].startswith("Weekly revenue")
    assert "up 200 (+20%) since last time" in mail["html"]
    assert "3 new rows" in mail["html"] and "https://app.test/reports" in mail["html"]
    assert [a["filename"] for a in mail["attachments"]] == ["Total-revenue.xlsx", "By-region.xlsx"]
