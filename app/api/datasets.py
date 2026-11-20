"""Upload, list, inspect, preview and delete datasets. Everything is workspace-scoped."""

import logging
import shutil
import threading
import uuid
from datetime import date
from pathlib import Path
from typing import Annotated, Literal

import duckdb
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy import select
from starlette.concurrency import run_in_threadpool

from app import audit, learning
from app.api.deps import AppSettings, CurrentUser, DbSession, Today, not_found, workspace_scoped
from app.connectors.csv_xlsx import (
    IMAGE_MIME,
    SPREADSHEET_ENGINES,
    IngestError,
    IngestResult,
    Transcriber,
    detect_format,
    ingest_file,
    pdf_is_mostly_tables,
)
from app.connectors.profiling import preview_parquet, profile_parquet
from app.core.config import Settings
from app.documents.ingest import DOCUMENT_FORMATS, ingest_document, load_passages
from app.llm.base import LLMCall, LLMProvider, LLMUnavailableError
from app.llm.embeddings import get_embedder
from app.llm.factory import make_vision_provider
from app.models import Dataset, GlossaryTerm, UsageEvent, User
from app.query.compare import compare_versions, comparison_counts
from app.query.doctor import Issue, apply_fixes, diagnose, summarize
from app.query.retrieval import column_tag, topic_texts
from app.schemas.dataset import (
    ChangesOut,
    DatasetOut,
    DatasetSummary,
    DoctorOut,
    FixIn,
    GlossaryTermIn,
    GlossaryTermOut,
    IssueOut,
    PreviewOut,
    UploadOut,
    VersionOut,
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


def _ingest_sheet(
    settings: Settings,
    workspace_id: str,
    original: Path,
    fmt: str,
    sheet: str,
) -> tuple[str, "IngestResult"] | None:
    """Copy one sheet of a workbook into its own dataset folder + Parquet (no DB writes).
    Returns (dataset_id, ingest result), or None if the sheet is empty/unreadable."""
    storage = _storage(settings)
    dataset_id = str(uuid.uuid4())
    folder = storage.create_dir(workspace_id, dataset_id)
    try:
        src = folder / f"original.{fmt}"
        shutil.copy(original, src)
        table = ingest_file(src, fmt, folder / PARQUET_NAME, sheet=sheet)  # type: ignore[arg-type]
        storage.persist(workspace_id, dataset_id)
        return dataset_id, table
    except Exception:
        log.warning("Extra sheet %r skipped", sheet, exc_info=True)
        storage.delete(workspace_id, dataset_id)
        return None


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

    # A workbook with several sheets: make each other sheet its own dataset, so columns
    # that live on another sheet (e.g. Assignee on "Sheet2") are queryable too.
    extra_sheets: list[str] = []
    if table is not None and fmt in SPREADSHEET_ENGINES and mode != "document":
        base = (name or "").strip() or Path(filename).stem or "Untitled"
        for other in [s for s in table.sheet_names if s != table.sheet_name]:
            made = await run_in_threadpool(
                _ingest_sheet, settings, user.workspace_id, original, fmt, other
            )
            if made is None:
                continue
            sib_id, sib = made
            sib_ds = Dataset(
                id=sib_id,
                workspace_id=user.workspace_id,
                created_by_id=user.id,
                name=f"{base} - {other}"[:200],
                original_filename=filename[:255],
                file_format=fmt,
                sheet_name=sib.sheet_name,
                size_bytes=size,
                row_count=sib.profile.row_count,
                columns=[c.to_dict() for c in sib.profile.columns],
                kind="table",
                extraction=sib.extraction,
            )
            db.add(sib_ds)
            audit.record(db, user, "dataset.uploaded", target_type="dataset",
                         target_id=sib_id, name=sib_ds.name, file=filename[:255],
                         kind="table")  # fmt: skip
            db.commit()
            _prepare_search(settings, sib_ds)
            extra_sheets.append(other)

    return UploadOut(
        **DatasetOut.model_validate(ds).model_dump(),
        sheet_names=table.sheet_names if table else [],
        detected_encoding=table.encoding if table else None,
        extra_sheets=extra_sheets,
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


# --- versions: "what changed?" and the data doctor's fixes ------------------------------


def save_new_version(
    settings: Settings, db: DbSession, user: User, ds: Dataset, new_parquet: Path,
    *, original: Path | None = None, action: str,
) -> ChangesOut:  # fmt: skip
    """Make `new_parquet` the dataset's data (the old data is kept as the previous
    version) and say what changed."""
    storage = _storage(settings)
    try:
        storage.replace_data(ds.workspace_id, ds.id, new_parquet, original)
        current = storage.ensure_local(ds.workspace_id, ds.id)
        previous = storage.ensure_previous(ds.workspace_id, ds.id)
    except StorageError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    if current is None or previous is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Couldn't save the new version.")
    profile = profile_parquet(current)
    ds.row_count = profile.row_count
    ds.columns = [c.to_dict() for c in profile.columns]
    ds.version += 1
    audit.record(db, user, action, target_type="dataset", target_id=ds.id, version=ds.version)
    db.commit()
    _prepare_search(settings, ds)
    return ChangesOut(**comparison_counts(compare_versions(previous, current)))


@router.post("/{dataset_id}/versions", response_model=VersionOut)
async def upload_new_version(
    dataset_id: str,
    file: Annotated[UploadFile, File(description="The new version of the same table")],
    user: CurrentUser,
    db: DbSession,
    settings: AppSettings,
    vision: Annotated[LLMProvider | None, Depends(get_vision_llm)],
    sheet: Annotated[str | None, Form()] = None,
) -> VersionOut:
    """Replace a table's data with a newer file (e.g. this week's export). The previous
    version is kept, so you can ask "what changed?"."""
    ds = _get_dataset(db, user, dataset_id)
    if ds.kind != "table":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "New versions work for tables only.")
    filename = Path(file.filename or "").name
    fmt = detect_format(filename)
    if fmt is None:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Unsupported file type.")
    folder = _storage(settings).dataset_dir(ds.workspace_id, ds.id) / "incoming"
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    try:
        original = folder / f"original.{fmt}"
        size = await _save_upload(file, original, settings.max_upload_mb * 1024 * 1024, settings)
        if size == 0:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "The file is empty.")
        calls: list[LLMCall] = []
        transcribe = _transcriber(vision, calls) if fmt in IMAGE_MIME or fmt == "pdf" else None
        out = folder / PARQUET_NAME
        try:
            table = await run_in_threadpool(ingest_file, original, fmt, out, sheet=sheet,
                                            transcribe=transcribe)  # fmt: skip
        except IngestError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
        ds.original_filename = filename[:255]
        ds.file_format = fmt
        ds.sheet_name = table.sheet_name
        ds.size_bytes = size
        ds.extraction = table.extraction
        changes = await run_in_threadpool(
            save_new_version, settings, db, user, ds, out, original=original,
            action="dataset.new_version",
        )  # fmt: skip
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    return VersionOut(dataset=DatasetOut.model_validate(ds), changes=changes)


def run_doctor(settings: Settings, ds: Dataset, today: date) -> tuple[list[Issue], Path]:
    if ds.kind != "table":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "The data doctor checks tables.")
    path = require_parquet(settings, ds)
    return diagnose(path, ds.columns, today), path


@router.get("/{dataset_id}/doctor", response_model=DoctorOut)
def dataset_doctor(
    dataset_id: str, user: CurrentUser, db: DbSession, settings: AppSettings, today: Today
) -> DoctorOut:
    """Problems in the data (duplicates, spellings, blanks, ...), most important first."""
    ds = _get_dataset(db, user, dataset_id)
    issues, _ = run_doctor(settings, ds, today)
    return DoctorOut(dataset_id=ds.id, version=ds.version, summary=summarize(issues, ds.name),
                     issues=[IssueOut(**i.as_dict()) for i in issues])  # fmt: skip


def fix_dataset(
    settings: Settings, db: DbSession, user: User, ds: Dataset, today: date, issue_ids: list[str]
) -> tuple[list[Issue], ChangesOut]:
    """Apply the safe fixes for these issues (all fixable ones if empty) as a new version."""
    issues, path = run_doctor(settings, ds, today)
    chosen = [i for i in issues if i.fix and (not issue_ids or i.id in issue_ids)]
    if not chosen:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "There's nothing to fix automatically.")
    folder = _storage(settings).dataset_dir(ds.workspace_id, ds.id)
    out = folder / "fixed.parquet"
    columns = [str(c["name"]) for c in ds.columns]
    try:
        apply_fixes(path, out, [i.fix for i in chosen if i.fix], columns)
    except (ValueError, duckdb.Error) as exc:
        out.unlink(missing_ok=True)
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, f"Couldn't apply the fixes: {exc}"
        ) from exc
    changes = save_new_version(settings, db, user, ds, out, action="dataset.fixed")
    return chosen, changes


@router.post("/{dataset_id}/doctor/fix", response_model=VersionOut)
def apply_doctor_fixes(
    dataset_id: str, body: FixIn, user: CurrentUser, db: DbSession, settings: AppSettings,
    today: Today,
) -> VersionOut:  # fmt: skip
    ds = _get_dataset(db, user, dataset_id)
    _fixed, changes = fix_dataset(settings, db, user, ds, today, body.issue_ids)
    return VersionOut(dataset=DatasetOut.model_validate(ds), changes=changes)


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
