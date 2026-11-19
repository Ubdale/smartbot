"""What changed between versions, and the data doctor (diagnose + safe fixes)."""

import json
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import Any

import duckdb
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.chat import CHANGES_REQUEST, DOCTOR_REQUEST, FIX_REQUEST
from app.api.deps import get_today
from app.llm.factory import get_llm
from app.query.compare import compare_versions, describe_comparison
from app.query.doctor import apply_fixes, diagnose
from app.query.executor import clear_cache
from tests.conftest import Account, signup
from tests.fakes import FakeLLM

TODAY = date(2026, 11, 19)
V1 = """Issue key,Assignee,Status,Hours
DMA-1,Zarmeen Khan,Open,4
DMA-2,Sara Ahmed,Open,6
DMA-3,Sara Khan,Done,2
DMA-4,Ali Raza,Open,8
"""
V2 = """Issue key,Assignee,Status,Hours
DMA-1,Zarmeen Khan,Done,4
DMA-2,Sara Ahmed,Done,7
DMA-4,Ali Raza,Open,8
DMA-5,Areesha Idrees,Open,3
"""
MESSY = """Ticket ID,Assignee,Status,Amount,Created,Notes
T-1,Sara Ahmed,Open,100,2026-10-01,
T-2,sara ahmed ,Open,120,2026-10-02,
T-3,Sara  Ahmed,Done,90,2026-10-03,
T-4,Zarmeen Khan,Open,110,2026-10-04,
T-5,Zarmeen Kahn,Open,105,2026-10-05,
T-5,Ali Raza,Done,95,2026-10-06,
T-6,Ali Raza,Done,99000,2027-10-06,
T-7,Ali Raza,Open,101,2026-10-07,
T-7,Ali Raza,Open,101,2026-10-07,
"""


def _parquet(tmp: Path, name: str, csv: str) -> Path:
    src = tmp / f"{name}.csv"
    src.write_text(csv, encoding="utf-8")
    out = tmp / f"{name}.parquet"
    with duckdb.connect() as con:
        con.execute(f"COPY (SELECT * FROM read_csv_auto('{src.as_posix()}')) "
                    f"TO '{out.as_posix()}' (FORMAT PARQUET)")  # fmt: skip
    return out


def _profile(path: Path) -> list[dict[str, Any]]:
    from app.connectors.profiling import profile_parquet

    return [c.to_dict() for c in profile_parquet(path).columns]


def test_compare_by_key(tmp_path: Path) -> None:
    c = compare_versions(_parquet(tmp_path, "v1", V1), _parquet(tmp_path, "v2", V2))
    assert c.key == "Issue key"
    assert (c.added, c.removed, c.changed_rows) == (1, 1, 2)
    assert c.changes_by_column == {"Status": 2, "Hours": 1}
    assert c.transitions["Status"][0] == ("Open", "Done", 2)
    assert ["Added", "DMA-5", "", "", ""] in c.rows
    assert ["Removed", "DMA-3", "", "", ""] in c.rows
    assert ["Changed", "DMA-2", "Hours", "6", "7"] in c.rows
    text = describe_comparison(c, "Tasks")
    assert "1 new row, 1 removed, 2 changed" in text
    assert "Status (2; most often Open → Done, 2 times)" in text


def test_compare_without_a_key_uses_whole_rows(tmp_path: Path) -> None:
    before = _parquet(tmp_path, "a", "city,team\nLahore,A\nLahore,A\nKarachi,B\n")
    after = _parquet(tmp_path, "b", "city,team\nLahore,A\nKarachi,B\nQuetta,C\n")
    c = compare_versions(before, after)
    assert c.key is None
    assert (c.added, c.removed) == (1, 1)  # one Lahore copy gone, Quetta new


def test_doctor_finds_problems_and_fixes_the_safe_ones(tmp_path: Path) -> None:
    path = _parquet(tmp_path, "messy", MESSY)
    issues = {i.id: i for i in diagnose(path, _profile(path), TODAY)}
    assert issues["duplicates"].rows == 1
    assert issues["dup-id:Ticket ID"].rows == 2  # T-5 twice (different people), T-7 twice
    spelling = issues["spelling:Assignee"]
    assert spelling.fix is not None
    assert spelling.fix["mapping"] == {"sara ahmed ": "Sara Ahmed", "Sara  Ahmed": "Sara Ahmed",
                                       "Zarmeen Kahn": "Zarmeen Khan"}  # fmt: skip
    assert "spaces:Assignee" in issues
    assert "empty-columns" in issues and issues["empty-columns"].examples == ["Notes"]
    assert issues["future:Created"].rows == 1
    assert issues["outlier:Amount"].examples == ["99000"]
    assert "dup-id:Ticket ID" not in {i.id for i in issues.values() if i.fix}  # never auto

    out = tmp_path / "fixed.parquet"
    columns = [c["name"] for c in _profile(path)]
    apply_fixes(path, out, [i.fix for i in issues.values() if i.fix], columns)
    with duckdb.connect() as con:
        names = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM '{out.as_posix()}'").fetchall()]
        sql = f"SELECT Assignee, count(*) FROM '{out.as_posix()}' GROUP BY 1"
        people = dict(con.execute(sql).fetchall())
    assert "Notes" not in names
    assert people == {"Sara Ahmed": 3, "Zarmeen Khan": 2, "Ali Raza": 3}  # 1 duplicate gone


def test_healthy_table_has_no_issues(tmp_path: Path) -> None:
    path = _parquet(tmp_path, "ok", V1)
    assert diagnose(path, _profile(path), TODAY) == []


def test_intents() -> None:
    for q in [
        "what changed?",
        "What's new since last week?",
        "what has changed",
        "compare with the previous version",
        "show me the differences since last upload",
    ]:
        assert CHANGES_REQUEST.search(q), q
    for q in [
        "check my data",
        "any problems in this file?",
        "find duplicates",
        "data quality",
        "is my data clean? check the sheet",
    ]:
        assert DOCTOR_REQUEST.search(q), q
    for q in ["fix them", "please fix it", "fix all", "clean it"]:
        assert FIX_REQUEST.search(q), q
    for q in [
        "how many new customers this month",
        "revenue by region",
        "fix the revenue numbers by month",
    ]:
        assert not CHANGES_REQUEST.search(q) and not FIX_REQUEST.search(q), q
        assert not DOCTOR_REQUEST.search(q), q


# --- through the API and chat ---------------------------------------------------------


@pytest.fixture(autouse=True)
def _setup(app: FastAPI) -> Iterator[None]:
    clear_cache()
    app.dependency_overrides[get_today] = lambda: TODAY
    yield
    clear_cache()


def test_versions_doctor_and_fixes_in_chat(app: FastAPI, client: TestClient) -> None:
    acct: Account = signup(client, "owner@acme.com", workspace="Acme")
    ds = client.post("/datasets", headers=acct.headers,
                     files={"file": ("tasks.csv", V1.encode())}).json()["id"]  # fmt: skip
    plan = {"metric": "count", "group_by": ["Status"], "chart": "bar"}
    app.dependency_overrides[get_llm] = lambda: FakeLLM([plan], summary="Mostly open.")

    def ask(q: str, conv: str | None = None) -> dict[str, Any]:
        body = {"question": q, **({"conversation_id": conv} if conv else {"dataset_id": ds})}
        resp = client.post("/chat/ask", headers=acct.headers, json=body)
        assert resp.status_code == 200, resp.text
        out: dict[str, Any] = resp.json()
        return out

    first = ask("tasks by status")
    conv = first["conversation_id"]
    reply = ask("what changed?", conv)["reply"]
    assert "only one version so far" in reply["content"]

    resp = client.post(f"/datasets/{ds}/versions", headers=acct.headers,
                       files={"file": ("tasks-week2.csv", V2.encode())})  # fmt: skip
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["dataset"]["version"] == 2 and body["dataset"]["row_count"] == 4
    assert body["changes"] == {"key": "Issue key", "added": 1, "removed": 1, "changed": 2,
                               "rows_before": 4, "rows_after": 4}  # fmt: skip

    # Proof rows of an answer from version 1 are refused, not silently wrong.
    old = client.get(f"/chat/messages/{first['reply']['id']}/evidence", headers=acct.headers)
    assert old.status_code == 409

    reply = ask("what changed since last week?", conv)["reply"]
    assert "1 new row, 1 removed, 2 changed" in reply["content"]
    assert ["Changed", "DMA-1", "Status", "Open", "Done"] in reply["answer"]["table"]["rows"]

    # The doctor, then "fix them": a new version with the fixes, previous kept.
    messy = MESSY.replace("Ticket ID", "Issue key").replace("T-", "DMA-")
    client.post(f"/datasets/{ds}/versions", headers=acct.headers,
                files={"file": ("messy.csv", messy.encode())})  # fmt: skip
    reply = ask("check my data", conv)["reply"]
    doctor = reply["answer"]["doctor"]
    assert doctor["version"] == 3
    assert {i["id"] for i in doctor["issues"]} >= {"duplicates", "spelling:Assignee"}

    reply = ask("fix them", conv)["reply"]
    assert "Saved as version 4" in reply["content"]
    reply = ask("what changed?", conv)["reply"]
    # IDs repeat in this file, so rows are compared whole: the old spellings are
    # "removed" rows and the tidied ones "added".
    rows = reply["answer"]["table"]["rows"]
    assert any(r[0] == "Removed" and "sara ahmed " in str(r[4]) for r in rows)
    assert any(r[0] == "Added" and "Assignee: Sara Ahmed" in str(r[4]) for r in rows)

    # The REST endpoints too.
    report = client.get(f"/datasets/{ds}/doctor", headers=acct.headers).json()
    assert all(i["id"] != "spelling:Assignee" for i in report["issues"])  # fixed
    other = signup(client, "spy@other.com", workspace="Other")
    assert client.get(f"/datasets/{ds}/doctor", headers=other.headers).status_code == 404
    assert json.dumps(report)  # serialisable
