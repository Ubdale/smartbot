"""Upload, list, inspect, preview and delete datasets. Everything is workspace-scoped."""

import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy import select
from starlette.concurrency import run_in_threadpool

from app.api.deps import AppSettings, CurrentUser, DbSession, not_found, workspace_scoped
from app.connectors.csv_xlsx import IngestError, detect_format, ingest_file
from app.connectors.profiling import preview_parquet
from app.core.config import Settings
from app.models import Dataset, User
from app.schemas.dataset import DatasetOut, DatasetSummary, PreviewOut, UploadOut
from app.storage import PARQUET_NAME, DatasetStorage, StorageError, get_storage

router = APIRouter(prefix="/datasets", tags=["datasets"])

CHUNK = 1024 * 1024
PREVIEW_MAX = 200


def _storage(settings: Settings) -> DatasetStorage:
    return get_storage(settings)


def require_parquet(settings: Settings, ds: Dataset) -> Path:
    """The dataset's local Parquet file (fetched from remote storage if needed)."""
    try:
        path = _storage(settings).ensure_local(ds.workspace_id, ds.id)
    except StorageError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    if path is None:
        raise HTTPException(status.HTTP_410_GONE, "The dataset's data file is missing.")
    return path


def _too_large(settings: Settings) -> HTTPException:
    return HTTPException(
        status.HTTP_413_CONTENT_TOO_LARGE,
        f"File is larger than the {settings.max_upload_mb} MB limit.",
    )


async def _save_upload(upload: UploadFile, dest: Path, max_bytes: int, settings: Settings) -> int:
    size = 0
    with dest.open("wb") as fh:
        while chunk := await upload.read(CHUNK):
            size += len(chunk)
            if size > max_bytes:
                raise _too_large(settings)
            fh.write(chunk)
    return size


def _get_dataset(db: DbSession, user: User, dataset_id: str) -> Dataset:
    stmt = workspace_scoped(select(Dataset).where(Dataset.id == dataset_id), Dataset, user)
    ds = db.scalar(stmt)
    if ds is None:
        raise not_found("Dataset")
    return ds


@router.post("", response_model=UploadOut, status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    file: Annotated[
        UploadFile, File(description="A table file: csv, xlsx, xls, ods, json, pdf...")
    ],
    user: CurrentUser,
    db: DbSession,
    settings: AppSettings,
    name: Annotated[str | None, Form(max_length=200)] = None,
    sheet: Annotated[
        str | None, Form(description="Spreadsheet sheet or PDF table (default: first/biggest)")
    ] = None,
) -> UploadOut:
    # BodySizeLimitMiddleware already refused bodies far over the limit before parsing;
    # this enforces the exact per-file limit while copying.
    max_bytes = settings.max_upload_mb * 1024 * 1024

    filename = Path(file.filename or "").name
    fmt = detect_format(filename)
    if fmt is None:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            "Unsupported file type. Use .csv, .tsv, .txt, .xlsx, .xls, .ods, .json, .parquet or "
            ".pdf.",
        )

    storage = _storage(settings)
    dataset_id = str(uuid.uuid4())
    folder = storage.create_dir(user.workspace_id, dataset_id)
    try:
        original = folder / f"original.{fmt}"
        size = await _save_upload(file, original, max_bytes, settings)
        if size == 0:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "The file is empty.")
        try:
            # CPU-bound (parsing, Parquet, profiling): keep the event loop free for others.
            result = await run_in_threadpool(
                ingest_file, original, fmt, folder / PARQUET_NAME, sheet=sheet
            )
        except IngestError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
        try:
            await run_in_threadpool(storage.persist, user.workspace_id, dataset_id)
        except StorageError as exc:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc

        ds = Dataset(
            id=dataset_id,
            workspace_id=user.workspace_id,
            created_by_id=user.id,
            name=(name or "").strip() or Path(filename).stem or "Untitled",
            original_filename=filename[:255],
            file_format=fmt,
            sheet_name=result.sheet_name,
            size_bytes=size,
            row_count=result.profile.row_count,
            columns=[c.to_dict() for c in result.profile.columns],
        )
        db.add(ds)
        db.commit()
    except BaseException:
        db.rollback()
        storage.delete(user.workspace_id, dataset_id)
        raise

    return UploadOut(
        **DatasetOut.model_validate(ds).model_dump(),
        sheet_names=result.sheet_names,
        detected_encoding=result.encoding,
    )


@router.get("", response_model=list[DatasetSummary])
def list_datasets(user: CurrentUser, db: DbSession) -> list[Dataset]:
    stmt = workspace_scoped(select(Dataset), Dataset, user).order_by(Dataset.created_at.desc())
    return list(db.scalars(stmt))


@router.get("/{dataset_id}", response_model=DatasetOut)
def get_dataset(dataset_id: str, user: CurrentUser, db: DbSession) -> Dataset:
    return _get_dataset(db, user, dataset_id)


@router.get("/{dataset_id}/preview", response_model=PreviewOut)
def preview_dataset(
    dataset_id: str,
    user: CurrentUser,
    db: DbSession,
    settings: AppSettings,
    limit: Annotated[int, Query(ge=1, le=PREVIEW_MAX)] = 50,
) -> PreviewOut:
    ds = _get_dataset(db, user, dataset_id)
    columns, rows = preview_parquet(require_parquet(settings, ds), limit)
    return PreviewOut(dataset_id=ds.id, columns=columns, rows=rows, row_count=ds.row_count)


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_dataset(
    dataset_id: str, user: CurrentUser, db: DbSession, settings: AppSettings
) -> None:
    ds = _get_dataset(db, user, dataset_id)
    db.delete(ds)
    db.commit()
    _storage(settings).delete(ds.workspace_id, ds.id)
