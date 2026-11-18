"""STORAGE_BACKEND=s3: files survive a wiped local disk (Render free plan)."""

import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import boto3
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from moto import mock_aws
from pydantic import SecretStr

from app.core.config import Settings, get_settings
from app.storage import _s3_client
from tests.conftest import Account, signup

SAMPLE = Path(__file__).resolve().parent.parent / "sample_data"
BUCKET = "askdata-test"


@pytest.fixture
def s3(app: FastAPI, settings: Settings) -> Iterator[Any]:
    _s3_client.cache_clear()
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        s3_settings = settings.model_copy(
            update={
                "storage_backend": "s3",
                "s3_bucket": BUCKET,
                "s3_prefix": "test/",
                "aws_access_key_id": SecretStr("testing"),
                "aws_secret_access_key": SecretStr("testing"),
            }
        )
        app.dependency_overrides[get_settings] = lambda: s3_settings
        yield client
    _s3_client.cache_clear()


def keys(client: Any) -> list[str]:
    return sorted(o["Key"] for o in client.list_objects_v2(Bucket=BUCKET).get("Contents", []))


def upload(client: TestClient, acct: Account) -> dict[str, Any]:
    content = (SAMPLE / "hr_employees.csv").read_bytes()
    resp = client.post("/datasets", headers=acct.headers, files={"file": ("hr.csv", content)})
    assert resp.status_code == 201, resp.text
    return dict(resp.json())


def test_upload_is_stored_remotely(s3: Any, client: TestClient) -> None:
    acct = signup(client, "s3@acme.com")
    ds = upload(client, acct)
    prefix = f"test/workspaces/{acct.workspace['id']}/datasets/{ds['id']}/"
    assert keys(s3) == [prefix + "data.parquet", prefix + "original.csv"]


def test_wiped_disk_is_restored_from_storage(
    s3: Any, client: TestClient, settings: Settings
) -> None:
    acct = signup(client, "s3@acme.com")
    ds = upload(client, acct)
    shutil.rmtree(settings.data_dir / "workspaces")  # what a Render restart does

    prev = client.get(f"/datasets/{ds['id']}/preview?limit=3", headers=acct.headers)
    assert prev.status_code == 200
    assert prev.json()["rows"][0][0] == "E0001"
    local = (
        settings.data_dir / "workspaces" / acct.workspace["id"] / "datasets" / ds["id"]
    ) / "data.parquet"
    assert local.exists()  # cached again
    assert not list(local.parent.glob("*.part"))  # no half-written temp files left


def test_delete_removes_remote_objects(s3: Any, client: TestClient) -> None:
    acct = signup(client, "s3@acme.com")
    ds = upload(client, acct)
    other = upload(client, acct)
    assert client.delete(f"/datasets/{ds['id']}", headers=acct.headers).status_code == 204
    assert all(other["id"] in k for k in keys(s3))
    assert len(keys(s3)) == 2


def test_missing_everywhere_is_410(s3: Any, client: TestClient, settings: Settings) -> None:
    acct = signup(client, "s3@acme.com")
    ds = upload(client, acct)
    shutil.rmtree(settings.data_dir / "workspaces")
    for k in keys(s3):
        s3.delete_object(Bucket=BUCKET, Key=k)
    assert client.get(f"/datasets/{ds['id']}/preview", headers=acct.headers).status_code == 410


def test_storage_outage_fails_upload_cleanly(
    s3: Any, client: TestClient, settings: Settings
) -> None:
    acct = signup(client, "s3@acme.com")
    s3.delete_bucket(Bucket=BUCKET)  # every upload will now fail
    content = (SAMPLE / "hr_employees.csv").read_bytes()
    resp = client.post("/datasets", headers=acct.headers, files={"file": ("hr.csv", content)})
    assert resp.status_code == 503
    assert "storage" in resp.json()["detail"]
    assert client.get("/datasets", headers=acct.headers).json() == []
    ws_dir = settings.data_dir / "workspaces" / acct.workspace["id"] / "datasets"
    assert not ws_dir.exists() or list(ws_dir.iterdir()) == []


def test_s3_settings_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="S3_BUCKET, AWS_ACCESS_KEY_ID"):
        Settings(_env_file=None, storage_backend="s3")
    # Neon hands out AWS_ENDPOINT_URL_S3; it is accepted as the endpoint.
    monkeypatch.setenv("AWS_ENDPOINT_URL_S3", "https://br-x.storage.neon.tech")
    s = Settings(_env_file=None)
    assert s.s3_endpoint_url == "https://br-x.storage.neon.tech"
