"""Upload, list, inspect, preview and delete datasets. Everything is workspace-scoped."""

import logging
import threading
import uuid
from pathlib import Path
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy import select
from starlette.concurrency import run_in_threadpool

from app import audit, learning
from app.api.deps import AppSettings, CurrentUser, DbSession, not_found, workspace_scoped
from app.connectors.csv_xlsx import (
    IMAGE_MIME,
    IngestError,
    Transcriber,
    detect_format,
    ingest_file,
    pdf_is_mostly_tables,
)
from app.connectors.profiling import preview_parquet
from app.core.config import Settings
from app.documents.ingest import DOCUMENT_FORMATS, ingest_document, load_passages
from app.llm.base import LLMCall, LLMProvider, LLMUnavailableError
from app.llm.embeddings import get_embedder
from app.llm.factory import make_vision_provider
from app.models import Dataset, GlossaryTerm, UsageEvent, User
from app.query.retrieval import column_tag, topic_texts
from app.schemas.dataset import (
    DatasetOut,
    DatasetSummary,
    GlossaryTermIn,
    GlossaryTermOut,
    PreviewOut,
    UploadOut,
)
from app.storage import PARQUET_NAME, DatasetStorage, StorageError, get_storage
from app.vectors import get_vectors

log = logging.getLogger(__name__)
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


def _prepare_search(settings: Settings, ds: Dataset) -> None:
    """Embed the new dataset's texts in the background, so search by meaning is ready
    by the first question. Optional: keyword search works meanwhile and without it."""
    embedder = get_embedder(settings)
    if embedder is None:
        return
    workspace_id, dataset_id, kind = ds.workspace_id, ds.id, ds.kind
    text_columns = [str(c["name"]) for c in ds.columns if c.get("kind") == "text"]

    def run() -> None:
        try:
            path = _storage(settings).ensure_local(workspace_id, dataset_id)
            if path is None:
                return
            if kind == "document":
                texts = [p.text for p in load_passages(path)]
                get_vectors(embedder, path, "passages", texts, wait=True)
                return
            for column in text_columns:
                column_texts = topic_texts(path, column)
                if column_texts:
                    get_vectors(embedder, path, column_tag(column), column_texts, wait=True)
        except Exception:  # never let background indexing affect anything else
            log.warning("Preparing search for dataset %s failed", dataset_id, exc_info=True)

    threading.Thread(target=run, name="prepare-search", daemon=True).start()


def get_vision_llm(settings: AppSettings) -> LLMProvider | None:
    """The model that reads tables from images; None when none is configured."""
    try:
        return make_vision_provider(settings)
    except LLMUnavailableError:
        return None


def _transcriber(vision: LLMProvider | None, calls: list[LLMCall]) -> Transcriber | None:
    if vision is None:
        return None

    def transcribe(image: bytes, mime: str) -> list[list[str]]:
        try:
            rows, call = vision.transcribe_table(image, mime)
        except LLMUnavailableError as exc:
            raise IngestError(str(exc)) from exc
        calls.append(call)
        return rows

    return transcribe


Mode = Literal["auto", "table", "document"]


def choose_kind(fmt: str, mode: Mode, path: Path) -> Literal["table", "document", "either"]:
    """Table (answers computed with SQL) or document (answers quoted from text)?"""
    if mode == "document":
        if fmt not in DOCUMENT_FORMATS:
            raise IngestError("Only PDF, Word, text and Markdown files can be read as documents.")
        return "document"
    if fmt in ("docx", "md"):
        if mode == "table":
            raise IngestError(f".{fmt} files are read as documents, not tables.")
        return "document"
    if mode == "table":
        return "table"
    if fmt == "pdf":
        return "table" if pdf_is_mostly_tables(path) else "document"
    if fmt == "txt":
        return "either"  # a delimited table if it parses as one, otherwise prose
    return "table"


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
    vision: Annotated[LLMProvider | None, Depends(get_vision_llm)],
    name: Annotated[str | None, Form(max_length=200)] = None,
    sheet: Annotated[
        str | None, Form(description="Spreadsheet sheet or PDF table (default: first/biggest)")
    ] = None,
    mode: Annotated[
        Mode, Form(description="auto, table (computed answers) or document (quoted answers)")
    ] = "auto",
) -> UploadOut:
    # BodySizeLimitMiddleware already refused bodies far over the limit before parsing;
    # this enforces the exact per-file limit while copying.
    max_bytes = settings.max_upload_mb * 1024 * 1024

    filename = Path(file.filename or "").name
    fmt = detect_format(filename)
    if fmt is None:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            "Unsupported file type. Use .csv, .tsv, .txt, .xlsx, .xls, .ods, .json, .parquet, "
            ".pdf, .docx, .md or an image (.png, .jpg, .webp).",
        )

    storage = _storage(settings)
    dataset_id = str(uuid.uuid4())
    folder = storage.create_dir(user.workspace_id, dataset_id)
    try:
        original = folder / f"original.{fmt}"
        size = await _save_upload(file, original, max_bytes, settings)
        if size == 0:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "The file is empty.")
        vision_calls: list[LLMCall] = []
        needs_vision = fmt in IMAGE_MIME or fmt == "pdf"  # images; scanned PDFs
        transcribe = _transcriber(vision, vision_calls) if needs_vision else None
        out = folder / PARQUET_NAME
        try:
            # CPU-bound (parsing, Parquet, profiling): keep the event loop free for others.
            kind = await run_in_threadpool(choose_kind, fmt, mode, original)
            table = None
            if kind in ("table", "either"):
                try:
                    table = await run_in_threadpool(
                        ingest_file, original, fmt, out, sheet=sheet, transcribe=transcribe
                    )
                except IngestError:
                    if kind == "table":
                        raise
                if kind == "either" and table is not None and len(table.profile.columns) < 2:
                    table = None  # one column of lines: it's prose, not a table
            document = None
            if table is None:
                out.unlink(missing_ok=True)
                document = await run_in_threadpool(ingest_document, original, fmt, out)
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
            sheet_name=table.sheet_name if table else None,
            size_bytes=size,
            row_count=table.profile.row_count if table else (document.passages if document else 0),
            columns=[c.to_dict() for c in table.profile.columns] if table else [],
            kind="table" if table else "document",
            extraction=table.extraction if table else None,
        )
        db.add(ds)
        audit.record(db, user, "dataset.uploaded", target_type="dataset",
                     target_id=dataset_id, name=ds.name, file=filename[:255],
                     kind=ds.kind)  # fmt: skip
        for call in vision_calls:
            db.add(
                UsageEvent(
                    workspace_id=user.workspace_id,
                    user_id=user.id,
                    purpose=call.purpose,
                    provider=call.provider,
                    model=call.model,
                    prompt_tokens=call.prompt_tokens,
                    completion_tokens=call.completion_tokens,
                    latency_ms=call.latency_ms,
                    success=call.success,
                )
            )
        db.commit()
    except BaseException:
        db.rollback()
        storage.delete(user.workspace_id, dataset_id)
        raise

    _prepare_search(settings, ds)
    return UploadOut(
        **DatasetOut.model_validate(ds).model_dump(),
        sheet_names=table.sheet_names if table else [],
        detected_encoding=table.encoding if table else None,
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
    audit.record(db, user, "dataset.deleted", target_type="dataset", target_id=ds.id,
                 name=ds.name)  # fmt: skip
    db.delete(ds)
    db.commit()
    _storage(settings).delete(ds.workspace_id, ds.id)


# --- the team's vocabulary for a file ("sales" means revenue) ---------------------------


@router.get("/{dataset_id}/glossary", response_model=list[GlossaryTermOut])
def list_glossary(dataset_id: str, user: CurrentUser, db: DbSession) -> list[GlossaryTerm]:
    ds = _get_dataset(db, user, dataset_id)
    stmt = select(GlossaryTerm).where(
        GlossaryTerm.workspace_id == user.workspace_id, GlossaryTerm.dataset_id == ds.id
    )
    return list(db.scalars(stmt.order_by(GlossaryTerm.term)))


@router.post("/{dataset_id}/glossary", response_model=list[GlossaryTermOut])
def add_glossary_term(
    dataset_id: str, body: GlossaryTermIn, user: CurrentUser, db: DbSession
) -> list[GlossaryTerm]:
    ds = _get_dataset(db, user, dataset_id)
    learning.remember_term(
        db,
        workspace_id=user.workspace_id,
        dataset_id=ds.id,
        user_id=user.id,
        d=learning.Definition(body.term.strip(), body.meaning.strip()),
    )
    db.commit()
    return list_glossary(dataset_id, user, db)


@router.delete("/{dataset_id}/glossary/{term_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_glossary_term(dataset_id: str, term_id: str, user: CurrentUser, db: DbSession) -> None:
    ds = _get_dataset(db, user, dataset_id)
    term = db.scalar(
        select(GlossaryTerm).where(
            GlossaryTerm.id == term_id,
            GlossaryTerm.workspace_id == user.workspace_id,
            GlossaryTerm.dataset_id == ds.id,
        )
    )
    if term is None:
        raise not_found("Term")
    db.delete(term)
    db.commit()
