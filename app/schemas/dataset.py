from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

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
    kind: str  # "table" | "document"
    extraction: str | None  # "ai_vision": read from an image by AI, please check
    created_at: datetime


class DatasetOut(DatasetSummary):
    columns: list[ColumnOut]


class UploadOut(DatasetOut):
    sheet_names: list[str]  # all sheets in the workbook (empty for CSV)
    detected_encoding: str | None
    # Other sheets of a workbook that were added as their own datasets.
    extra_sheets: list[str] = []


class ChangesOut(BaseModel):
    """What a new version changed, compared with the one before it."""

    key: str | None  # the column rows were matched by
    added: int
    removed: int
    changed: int
    rows_before: int
    rows_after: int


class VersionOut(BaseModel):
    dataset: DatasetOut
    changes: ChangesOut


class IssueOut(BaseModel):
    id: str
    severity: str  # high | medium | low
    title: str
    column: str | None
    rows: int
    detail: str
    examples: list[str]
    fixable: bool
    fix_label: str | None


class DoctorOut(BaseModel):
    dataset_id: str
    version: int
    summary: str
    issues: list[IssueOut]


class FixIn(BaseModel):
    # Issue ids from the doctor's report; empty = every fixable issue.
    issue_ids: list[str] = Field(default_factory=list, max_length=100)


class PreviewOut(BaseModel):
    dataset_id: str
    columns: list[str]
    rows: list[list[Any]]
    row_count: int  # total rows in the dataset, not just this preview


class GlossaryTermIn(BaseModel):
    term: str = Field(min_length=1, max_length=100)
    meaning: str = Field(min_length=1, max_length=300)


class GlossaryTermOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    term: str
    meaning: str
    created_at: datetime
