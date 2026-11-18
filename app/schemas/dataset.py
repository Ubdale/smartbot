from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from app.connectors.profiling import ColumnKind


class ColumnOut(BaseModel):
    name: str
    type: str
    kind: ColumnKind
    null_count: int
    distinct_count: int
    samples: list[Any]
    min: Any = None
    max: Any = None


class DatasetSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    original_filename: str
    file_format: str
    sheet_name: str | None
    size_bytes: int
    row_count: int
    version: int
    created_at: datetime


class DatasetOut(DatasetSummary):
    columns: list[ColumnOut]


class UploadOut(DatasetOut):
    sheet_names: list[str]  # all sheets in the workbook (empty for CSV)
    detected_encoding: str | None


class PreviewOut(BaseModel):
    dataset_id: str
    columns: list[str]
    rows: list[list[Any]]
    row_count: int  # total rows in the dataset, not just this preview
