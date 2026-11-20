import io
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from tests.conftest import Account, signup

SAMPLE = Path(__file__).resolve().parent.parent / "sample_data"


@pytest.fixture
def acct(client: TestClient) -> Account:
    return signup(client, "owner@acme.com", workspace="Acme")


def upload(client: TestClient, acct: Account, filename: str, content: bytes, **form: str) -> Any:
    return client.post(
        "/datasets",
        headers=acct.headers,
        files={"file": (filename, content, "application/octet-stream")},
        data=form,
    )


def xlsx_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)
    return buf.getvalue()


def cols(body: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {c["name"]: c for c in body["columns"]}


# --- CSV ---------------------------------------------------------------------------


def test_upload_sample_sales_csv(client: TestClient, acct: Account) -> None:
    resp = upload(client, acct, "sales_orders.csv", (SAMPLE / "sales_orders.csv").read_bytes())
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["name"] == "sales_orders"
    assert body["file_format"] == "csv"
    assert body["row_count"] == 3000
    assert body["detected_encoding"] == "utf-8"

    c = cols(body)
    assert c["order_date"]["kind"] == "date"
    assert c["order_date"]["min"] == "2024-01-01"
    assert c["revenue"]["kind"] == "number"
    assert c["quantity"]["kind"] == "number"
    assert c["region"]["kind"] == "text"
    assert c["region"]["distinct_count"] == pytest.approx(4, abs=1)
    assert len(c["region"]["samples"]) == 3
    assert c["region"]["null_count"] == 0


def test_hr_csv_nulls_and_booleans(client: TestClient, acct: Account) -> None:
    resp = upload(client, acct, "hr.csv", (SAMPLE / "hr_employees.csv").read_bytes())
    assert resp.status_code == 201, resp.text
    c = cols(resp.json())
    assert c["termination_date"]["kind"] == "date"
    assert c["termination_date"]["null_count"] > 0
    assert c["is_remote"]["kind"] == "boolean"
    assert c["hire_date"]["kind"] == "date"


def test_windows_cp1252_csv_is_decoded(client: TestClient, acct: Account) -> None:
    content = "city,amount\nZürich,10\nCafé Société,20\n".encode("cp1252")
    resp = upload(client, acct, "legacy.csv", content)
    assert resp.status_code == 201, resp.text
    assert resp.json()["detected_encoding"] == "cp1252"
    preview = client.get(f"/datasets/{resp.json()['id']}/preview", headers=acct.headers).json()
    assert [r[0] for r in preview["rows"]] == ["Zürich", "Café Société"]


def test_utf8_bom_is_stripped(client: TestClient, acct: Account) -> None:
    resp = upload(client, acct, "bom.csv", "﻿name,value\na,1\n".encode())
    assert resp.status_code == 201, resp.text
    assert [c["name"] for c in resp.json()["columns"]] == ["name", "value"]


def test_messy_headers_are_cleaned(client: TestClient, acct: Account) -> None:
    content = b'"  Total   Sales ",Region,region,"say ""hi"""\n1,a,b,c\n'
    resp = upload(client, acct, "messy.csv", content)
    assert resp.status_code == 201, resp.text
    names = [c["name"] for c in resp.json()["columns"]]
    assert names[0] == "Total Sales"
    assert names[3] == 'say "hi"'
    assert len({n.lower() for n in names}) == len(names)  # unique, case-insensitively


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"", "empty"),
        (b"   \n\n", "empty"),
        (b"a,b,c\n", "no data rows"),
    ],
)
def test_bad_csv_rejected(client: TestClient, acct: Account, content: bytes, message: str) -> None:
    resp = upload(client, acct, "bad.csv", content)
    assert resp.status_code == 400
    assert message in resp.json()["detail"]


def test_unsupported_extension(client: TestClient, acct: Account) -> None:
    for name in ("deck.pptx", "data.exe", "data", "evil.csv.exe"):
        assert upload(client, acct, name, b"a,b\n1,2\n").status_code == 415


def test_too_large_rejected(app: FastAPI, client: TestClient, settings: Settings) -> None:
    acct = signup(client, "big@acme.com")
    app.dependency_overrides[get_settings] = lambda: settings.model_copy(
        update={"max_upload_mb": 1}
    )
    big = b"a,b\n" + b"123456789,abcdefghi\n" * 60_000  # ~1.2 MB
    resp = upload(client, acct, "big.csv", big)
    assert resp.status_code == 413
    assert client.get("/datasets", headers=acct.headers).json() == []


def test_huge_body_rejected_before_parsing(settings: Settings) -> None:
    """The middleware answers 413 from Content-Length alone, before reading the body."""
    import asyncio

    from app.core.limits import BodySizeLimitMiddleware

    reached_app = False

    async def inner(scope: Any, receive: Any, send: Any) -> None:
        nonlocal reached_app
        reached_app = True

    mw = BodySizeLimitMiddleware(inner, max_bytes=1000, path_prefixes=("/datasets",))
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        raise AssertionError("body must not be read")

    async def send(message: Any) -> None:
        sent.append(message)

    scope = {"type": "http", "path": "/datasets", "headers": [(b"content-length", b"5000000000")]}
    asyncio.run(mw(scope, receive, send))
    assert not reached_app
    assert sent[0]["status"] == 413


def test_chunked_body_over_limit_rejected() -> None:
    """Without Content-Length, the middleware counts bytes as they stream in."""
    import asyncio

    from app.core.limits import BodySizeLimitMiddleware

    async def inner(scope: Any, receive: Any, send: Any) -> None:
        while (await receive()).get("more_body"):
            pass
        raise AssertionError("should have been stopped")

    mw = BodySizeLimitMiddleware(inner, max_bytes=1000, path_prefixes=("/datasets",))
    chunks = iter([{"type": "http.request", "body": b"x" * 600, "more_body": True}] * 3)
    sent: list[dict[str, Any]] = []

    async def receive() -> Any:
        return next(chunks)

    async def send(message: Any) -> None:
        sent.append(message)

    asyncio.run(mw({"type": "http", "path": "/datasets", "headers": []}, receive, send))
    assert sent[0]["status"] == 413


def test_failed_upload_leaves_no_files(
    client: TestClient, acct: Account, settings: Settings
) -> None:
    upload(client, acct, "bad.csv", b"a,b,c\n")
    ws_dir = settings.data_dir / "workspaces" / acct.workspace["id"] / "datasets"
    assert not ws_dir.exists() or list(ws_dir.iterdir()) == []


# --- XLSX ----------------------------------------------------------------------------


def test_xlsx_first_sheet_by_default(client: TestClient, acct: Account) -> None:
    df = pd.DataFrame(
        {
            "Month": pd.to_datetime(["2026-01-01", "2026-02-01"]),
            "Revenue": [1000.5, 2000.25],
            "Team": ["North", "South"],
        }
    )
    other = pd.DataFrame({"x": [1, 2, 3]})
    resp = upload(client, acct, "Budget 2026.xlsx", xlsx_bytes({"Summary": df, "Raw": other}))
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["name"] == "Budget 2026"
    assert body["sheet_name"] == "Summary"
    assert body["sheet_names"] == ["Summary", "Raw"]
    assert body["extra_sheets"] == ["Raw"]  # the other sheet became its own dataset
    assert body["row_count"] == 2
    # Both sheets are now selectable datasets.
    names = {d["name"] for d in client.get("/datasets", headers=acct.headers).json()}
    assert {"Budget 2026", "Budget 2026 - Raw"} <= names
    c = cols(body)
    assert c["Month"]["kind"] in ("date", "datetime")
    assert c["Revenue"]["kind"] == "number"
    assert c["Team"]["kind"] == "text"


def test_xlsx_choose_sheet(client: TestClient, acct: Account) -> None:
    content = xlsx_bytes({"A": pd.DataFrame({"x": [1]}), "B": pd.DataFrame({"y": [1, 2]})})
    resp = upload(client, acct, "book.xlsx", content, sheet="B", name="My data")
    assert resp.status_code == 201, resp.text
    assert resp.json()["sheet_name"] == "B"
    assert resp.json()["name"] == "My data"
    assert resp.json()["row_count"] == 2

    missing = upload(client, acct, "book.xlsx", content, sheet="Nope")
    assert missing.status_code == 400
    assert "Available: A, B" in missing.json()["detail"]


def test_xlsx_mixed_column_kept_as_text(client: TestClient, acct: Account) -> None:
    df = pd.DataFrame({"code": [101, "A-7", 103], "n": [1, 2, 3]})
    resp = upload(client, acct, "mixed.xlsx", xlsx_bytes({"S": df}))
    assert resp.status_code == 201, resp.text
    assert cols(resp.json())["code"]["kind"] == "text"
    preview = client.get(f"/datasets/{resp.json()['id']}/preview", headers=acct.headers).json()
    assert [r[0] for r in preview["rows"]] == ["101", "A-7", "103"]


def test_corrupt_xlsx_rejected(client: TestClient, acct: Account) -> None:
    resp = upload(client, acct, "broken.xlsx", b"this is not a zip file")
    assert resp.status_code == 400
    assert "spreadsheet" in resp.json()["detail"]


# --- list / get / preview / delete -----------------------------------------------------


def test_list_get_preview_delete(client: TestClient, acct: Account, settings: Settings) -> None:
    ds = upload(client, acct, "hr.csv", (SAMPLE / "hr_employees.csv").read_bytes()).json()

    listed = client.get("/datasets", headers=acct.headers).json()
    assert [d["id"] for d in listed] == [ds["id"]]
    assert "columns" not in listed[0]  # summaries only

    got = client.get(f"/datasets/{ds['id']}", headers=acct.headers).json()
    assert got["columns"] == ds["columns"]

    prev = client.get(f"/datasets/{ds['id']}/preview?limit=5", headers=acct.headers).json()
    assert prev["row_count"] == 400
    assert len(prev["rows"]) == 5
    assert prev["columns"][0] == "employee_id"
    assert prev["rows"][0][0] == "E0001"
    assert isinstance(prev["rows"][0][5], str)  # hire_date as ISO string

    too_many = client.get(f"/datasets/{ds['id']}/preview?limit=1000", headers=acct.headers)
    assert too_many.status_code == 422

    folder = settings.data_dir / "workspaces" / acct.workspace["id"] / "datasets" / ds["id"]
    assert (folder / "data.parquet").exists()
    assert client.delete(f"/datasets/{ds['id']}", headers=acct.headers).status_code == 204
    assert not folder.exists()
    assert client.get(f"/datasets/{ds['id']}", headers=acct.headers).status_code == 404


def test_preview_null_values_are_json_null(client: TestClient, acct: Account) -> None:
    resp = upload(client, acct, "n.csv", b"a,b\n1,\n,2.5\n")
    prev = client.get(f"/datasets/{resp.json()['id']}/preview", headers=acct.headers).json()
    assert prev["rows"] == [[1, None], [None, 2.5]]


# --- isolation -------------------------------------------------------------------------


def test_datasets_are_workspace_isolated(client: TestClient, settings: Settings) -> None:
    a = signup(client, "a@a-corp.com", workspace="A")
    b = signup(client, "b@b-corp.com", workspace="B")
    ds = upload(client, a, "secret.csv", b"salary\n100000\n").json()

    assert client.get("/datasets", headers=b.headers).json() == []
    for method, path in [
        ("GET", f"/datasets/{ds['id']}"),
        ("GET", f"/datasets/{ds['id']}/preview"),
        ("DELETE", f"/datasets/{ds['id']}"),
    ]:
        assert client.request(method, path, headers=b.headers).status_code == 404, path

    # A's files sit under A's workspace folder only, and survived B's delete attempt.
    folder = settings.data_dir / "workspaces" / a.workspace["id"] / "datasets" / ds["id"]
    assert (folder / "data.parquet").exists()
    assert client.get(f"/datasets/{ds['id']}", headers=a.headers).status_code == 200


@pytest.mark.parametrize("path", ["/datasets", "/datasets/x", "/datasets/x/preview"])
def test_dataset_routes_require_auth(client: TestClient, path: str) -> None:
    assert client.get(path).status_code == 401


def test_chunked_over_limit_through_real_app_is_413(app: FastAPI, client: TestClient) -> None:
    """End to end: FastAPI would turn the abort into a 400; the middleware makes it 413."""
    from app.core.limits import BodySizeLimitMiddleware

    acct = signup(client, "chunk@acme.com")
    small = FastAPI()
    small.router = app.router
    small.dependency_overrides = app.dependency_overrides
    small.add_middleware(BodySizeLimitMiddleware, max_bytes=50_000, path_prefixes=("/datasets",))

    def body() -> Any:
        boundary = b"--b0undary"
        yield (
            boundary + b'\r\nContent-Disposition: form-data; name="file"; filename="x.csv"\r\n\r\n'
        )
        for _ in range(20):
            yield b"a,b\n" + b"1,2\n" * 1000
        yield b"\r\n" + boundary + b"--\r\n"

    with TestClient(small) as c:
        resp = c.post(
            "/datasets",
            headers={**acct.headers, "content-type": "multipart/form-data; boundary=b0undary"},
            content=body(),
        )
    assert resp.status_code == 413


def test_asking_about_sheets_lists_them(app: Any, client: TestClient, acct: Account) -> None:
    from app.api.deps import get_today
    from app.llm.factory import get_llm
    from tests.fakes import FakeLLM

    app.dependency_overrides[get_today] = lambda: date(2026, 11, 19)
    app.dependency_overrides[get_llm] = lambda: FakeLLM([])  # intent answers without the model
    content = xlsx_bytes(
        {
            "Tasks": pd.DataFrame({"Task": ["a", "b"]}),
            "People": pd.DataFrame({"Assignee": ["Sara", "Ali", "Zara"]}),
        }
    )
    ds = upload(client, acct, "Project.xlsx", content).json()["id"]
    resp = client.post(
        "/chat/ask",
        headers=acct.headers,
        json={"question": "tell me number of sheets", "dataset_id": ds},
    )
    assert resp.status_code == 200, resp.text
    reply = resp.json()["reply"]
    assert "2 sheets" in reply["content"]
    sheet_names = {r[0] for r in reply["answer"]["table"]["rows"]}
    assert sheet_names == {"Tasks", "People"}
