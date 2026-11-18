"""Where dataset files live.

Layout (local disk):  {data_dir}/workspaces/{workspace_id}/datasets/{dataset_id}/
    original.csv|xlsx   the uploaded file, kept for re-ingestion
    data.parquet        canonical, cleaned copy that every query reads

Paths are built only from server-generated UUIDs, never from the uploaded
filename. An S3-compatible backend (e.g. Neon object storage) can replace this
class in hosted mode.
"""

import re
import shutil
from pathlib import Path

_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")

PARQUET_NAME = "data.parquet"


class DatasetStorage:
    def __init__(self, data_dir: Path) -> None:
        self.root = (data_dir / "workspaces").resolve()

    def dataset_dir(self, workspace_id: str, dataset_id: str) -> Path:
        for value in (workspace_id, dataset_id):
            if not _UUID.match(value):
                raise ValueError(f"not a server-generated id: {value!r}")
        path = (self.root / workspace_id / "datasets" / dataset_id).resolve()
        if not path.is_relative_to(self.root):  # defence in depth
            raise ValueError("path escapes storage root")
        return path

    def create_dir(self, workspace_id: str, dataset_id: str) -> Path:
        path = self.dataset_dir(workspace_id, dataset_id)
        path.mkdir(parents=True, exist_ok=False)
        return path

    def parquet_path(self, workspace_id: str, dataset_id: str) -> Path:
        return self.dataset_dir(workspace_id, dataset_id) / PARQUET_NAME

    def delete(self, workspace_id: str, dataset_id: str) -> None:
        shutil.rmtree(self.dataset_dir(workspace_id, dataset_id), ignore_errors=True)
