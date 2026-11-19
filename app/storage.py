"""Where dataset files live.

Local layout:  {data_dir}/workspaces/{workspace_id}/datasets/{dataset_id}/
    original.csv|xlsx   the uploaded file, kept for re-ingestion
    data.parquet        canonical, cleaned copy that every query reads

DuckDB always reads the local copy. With STORAGE_BACKEND=s3 the files are also
stored in S3-compatible object storage (e.g. Neon), and the local directory acts
as a cache: after a restart on a host with a temporary disk (Render's free plan),
a missing Parquet file is downloaded again on first use.

Paths and keys are built only from server-generated UUIDs, never from filenames.
"""

import logging
import os
import re
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from app.core.config import Settings

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client
    from mypy_boto3_s3.type_defs import ObjectIdentifierTypeDef

log = logging.getLogger(__name__)

_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")

PARQUET_NAME = "data.parquet"
PREVIOUS_NAME = "previous.parquet"  # the version before the latest, for "what changed?"


class StorageError(RuntimeError):
    """Remote storage failed. The message is safe to show users."""


class RemoteStore(Protocol):
    def upload(self, local: Path, key: str) -> None: ...
    def download(self, key: str, local: Path) -> bool: ...
    def delete_prefix(self, prefix: str) -> None: ...


class S3Store:
    def __init__(self, client: "S3Client", bucket: str, prefix: str = "") -> None:
        self._s3 = client
        self.bucket = bucket
        self.prefix = prefix

    def upload(self, local: Path, key: str) -> None:
        self._s3.upload_file(str(local), self.bucket, self.prefix + key)

    def download(self, key: str, local: Path) -> bool:
        from botocore.exceptions import ClientError

        local.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=local.parent, suffix=".part")
        os.close(fd)
        try:
            self._s3.download_file(self.bucket, self.prefix + key, tmp)
        except ClientError as exc:
            Path(tmp).unlink(missing_ok=True)
            if exc.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
                return False
            raise
        os.replace(tmp, local)  # atomic: concurrent readers never see a partial file
        return True

    def delete_prefix(self, prefix: str) -> None:
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix + prefix):
            keys: list[ObjectIdentifierTypeDef] = [
                {"Key": o["Key"]} for o in page.get("Contents", []) if "Key" in o
            ]
            if keys:
                self._s3.delete_objects(Bucket=self.bucket, Delete={"Objects": keys})


class DatasetStorage:
    def __init__(self, data_dir: Path, remote: RemoteStore | None = None) -> None:
        self.root = (data_dir / "workspaces").resolve()
        self.remote = remote

    # --- paths ------------------------------------------------------------------------

    def dataset_dir(self, workspace_id: str, dataset_id: str) -> Path:
        for value in (workspace_id, dataset_id):
            if not _UUID.match(value):
                raise ValueError(f"not a server-generated id: {value!r}")
        path = (self.root / workspace_id / "datasets" / dataset_id).resolve()
        if not path.is_relative_to(self.root):  # defence in depth
            raise ValueError("path escapes storage root")
        return path

    @staticmethod
    def _key(workspace_id: str, dataset_id: str, name: str = "") -> str:
        return f"workspaces/{workspace_id}/datasets/{dataset_id}/{name}"

    def create_dir(self, workspace_id: str, dataset_id: str) -> Path:
        path = self.dataset_dir(workspace_id, dataset_id)
        path.mkdir(parents=True, exist_ok=False)
        return path

    def parquet_path(self, workspace_id: str, dataset_id: str) -> Path:
        return self.dataset_dir(workspace_id, dataset_id) / PARQUET_NAME

    # --- lifecycle ----------------------------------------------------------------------

    def persist(self, workspace_id: str, dataset_id: str) -> None:
        """Copy a freshly ingested dataset's files to remote storage (no-op locally)."""
        if self.remote is None:
            return
        folder = self.dataset_dir(workspace_id, dataset_id)
        try:
            for f in sorted(folder.iterdir()):
                if f.is_file():
                    self.remote.upload(f, self._key(workspace_id, dataset_id, f.name))
        except Exception as exc:
            log.exception("upload to remote storage failed")
            raise StorageError("Couldn't save the file to storage. Please try again.") from exc

    def ensure_local(self, workspace_id: str, dataset_id: str) -> Path | None:
        """The local Parquet path, downloading it first if needed. None if it's gone."""
        path = self.parquet_path(workspace_id, dataset_id)
        if path.exists():
            return path
        if self.remote is None:
            return None
        try:
            found = self.remote.download(self._key(workspace_id, dataset_id, PARQUET_NAME), path)
        except Exception as exc:
            log.exception("download from remote storage failed")
            raise StorageError("Couldn't load the dataset from storage. Try again.") from exc
        return path if found else None

    def ensure_previous(self, workspace_id: str, dataset_id: str) -> Path | None:
        """The previous version's local Parquet path, or None if there's no previous one."""
        path = self.dataset_dir(workspace_id, dataset_id) / PREVIOUS_NAME
        if path.exists():
            return path
        if self.remote is None:
            return None
        try:
            found = self.remote.download(self._key(workspace_id, dataset_id, PREVIOUS_NAME), path)
        except Exception as exc:
            log.exception("download of previous version failed")
            raise StorageError("Couldn't load the previous version. Try again.") from exc
        return path if found else None

    def replace_data(
        self, workspace_id: str, dataset_id: str, new_parquet: Path, original: Path | None = None
    ) -> None:
        """Make `new_parquet` the current data; the current data becomes the previous
        version. `original`: the new uploaded file, replacing the old one."""
        folder = self.dataset_dir(workspace_id, dataset_id)
        current = folder / PARQUET_NAME
        if not current.exists() and self.ensure_local(workspace_id, dataset_id) is None:
            raise StorageError("The current version is missing.")
        os.replace(current, folder / PREVIOUS_NAME)
        os.replace(new_parquet, current)
        if original is not None:
            for old in folder.glob("original.*"):
                old.unlink(missing_ok=True)
            os.replace(original, folder / original.name)
        for stale in folder.glob("*.vec.npy"):  # search vectors of the old data
            stale.unlink(missing_ok=True)
        self.persist(workspace_id, dataset_id)

    def delete(self, workspace_id: str, dataset_id: str) -> None:
        shutil.rmtree(self.dataset_dir(workspace_id, dataset_id), ignore_errors=True)
        if self.remote is not None:
            try:
                self.remote.delete_prefix(self._key(workspace_id, dataset_id))
            except Exception:
                # The database row is already gone; orphaned objects are harmless.
                log.exception("remote delete failed for dataset %s", dataset_id)


@lru_cache(maxsize=4)
def _s3_client(endpoint: str | None, region: str, key_id: str, secret: str) -> "S3Client":
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=region,
        aws_access_key_id=key_id,
        aws_secret_access_key=secret,
        config=Config(
            s3={"addressing_style": "path"},  # widest support among S3-compatible stores
            # Newer boto3 adds CRC checksums by default, which some S3-compatible
            # services reject. Only send them where the API requires it.
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
            retries={"max_attempts": 3, "mode": "standard"},
        ),
    )


def get_storage(settings: Settings) -> DatasetStorage:
    if settings.storage_backend == "local":
        return DatasetStorage(settings.data_dir)
    bucket = settings.s3_bucket
    key_id = settings.aws_access_key_id
    secret = settings.aws_secret_access_key
    if not bucket or key_id is None or secret is None:  # Settings enforces this already
        raise StorageError("S3 storage is not configured.")
    client = _s3_client(
        settings.s3_endpoint_url,
        settings.aws_region,
        key_id.get_secret_value(),
        secret.get_secret_value(),
    )
    return DatasetStorage(settings.data_dir, S3Store(client, bucket, settings.s3_prefix))
