"""Ask questions about a dataset; list and read conversations."""

import json
import logging
import queue
import re
import threading
from collections.abc import Callable, Iterator
from datetime import UTC, date, datetime
from itertools import pairwise
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import Response, StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app import audit, learning
from app.api.datasets import _storage, fix_dataset, require_parquet, run_doctor
from app.api.deps import AppSettings, CurrentUser, DbSession, Today, not_found, workspace_scoped
from app.core import ratelimit
from app.core.config import Settings
from app.core.db import get_session_factory
from app.digests import describe_schedule, is_schedule_request, parse_schedule
from app.documents.answer import DocumentAnswer, answer_document_question
from app.export_formats import FORMATS, NothingToExportError
from app.exporting import _pretty, export_title
from app.llm.base import LLMCall, LLMProvider, LLMUnavailableError
from app.llm.embeddings import get_embedder
from app.llm.factory import get_llm, get_summary_llm
from app.models import Conversation, Dataset, Message, UsageEvent, User
from app.query.answer import Answer, answer_question
from app.query.codegen import pandas_code
from app.query.compare import compare_versions, describe_comparison
from app.query.doctor import summarize
from app.query.evidence import find_evidence
from app.query.executor import (
    PlanError,
    QueryTimeoutError,
    columns_from_profile,
    execute_plan,
    validate_plan,
)
from app.query.planner import history_messages
from app.query.presentation import choose_chart, describe_plan
from app.schemas.chat import (
    AskRequest,
    AskResponse,
    ConversationOut,
    ConversationSummary,
    EvidenceOut,
    FeedbackOut,
    FeedbackRequest,
    MessageOut,
)
from app.schemas.query_plan import QueryPlan
from app.styling import ExportStyle, describe_style, interpret_style

router = APIRouter(prefix="/chat", tags=["chat"])
log = logging.getLogger(__name__)

HISTORY_TURNS = 3


def _conversation(db: DbSession, user: User, conversation_id: str) -> Conversation:
    stmt = workspace_scoped(
        select(Conversation).where(
            Conversation.id == conversation_id, Conversation.user_id == user.id
        ),
        Conversation,
        user,
    )
    conv = db.scalar(stmt)
    if conv is None:
        raise not_found("Conversation")
    return conv


def _dataset(db: DbSession, user: User, dataset_id: str) -> Dataset:
    stmt = workspace_scoped(select(Dataset).where(Dataset.id == dataset_id), Dataset, user)
    ds = db.scalar(stmt)
    if ds is None:
        raise not_found("Dataset")
    return ds


AI_VISION_NOTE = (
    "This table was read from an image by AI. The calculation is exact, but check the "
    "dataset preview to make sure the numbers were read correctly."
)


def _answer_payload(a: Answer, ds: Dataset) -> dict[str, Any] | None:
    payload = _table_payload(a, ds)
    if payload is not None:
        payload["reused"] = a.reused
    return payload


def _table_payload(a: Answer, ds: Dataset) -> dict[str, Any] | None:
    if a.status != "answered" or a.result is None or a.validated is None:
        return None
    r, vp = a.result, a.validated
    dr = vp.date_range
    return {
        "table": {"columns": r.columns, "rows": r.rows, "truncated": r.truncated},
        "chart": a.chart.__dict__ if a.chart else None,
        "explanation": {
            "description": a.description or "",
            "sql": r.sql,
            "params": r.params,
            "date_range": (
                {"start": dr.start.isoformat(), "end": dr.last_day.isoformat(), "label": dr.label}
                if dr
                else None
            ),
            "plan": vp.plan.model_dump(mode="json"),
            "cached": r.cached,
            "elapsed_ms": r.elapsed_ms,
            "data_note": AI_VISION_NOTE if ds.extraction == "ai_vision" else None,
            "dataset_version": ds.version,
            # Table RAG: how names and topics were matched, and the matched rows (proof).
            "retrieval_notes": a.retrieval_notes,
            "topic_rows": {str(k): v[:5000] for k, v in a.topic_rows.items()},
        },
        "summary_source": a.summary_source,
    }


def _document_payload(doc: DocumentAnswer) -> dict[str, Any] | None:
    if not doc.passages:
        return None
    return {
        "type": "document",
        "grounded": doc.grounded,
        "sources": [
            {"n": i, "page": p.page, "text": p.text, "cited": i in doc.cited}
            for i, p in enumerate(doc.passages, start=1)
        ],
    }


@router.post("/ask", response_model=AskResponse)
def ask(
    body: AskRequest,
    user: CurrentUser,
    db: DbSession,
    settings: AppSettings,
    llm: Annotated[LLMProvider, Depends(get_llm)],
    summarizer: Annotated[LLMProvider, Depends(get_summary_llm)],
    today: Today,
) -> AskResponse:
    ratelimit.questions.check(user.id, settings.questions_per_minute)
    return run_ask(body, user, db, settings, llm, summarizer, today)


@router.post("/ask/stream")
def ask_stream(
    body: AskRequest,
    user: CurrentUser,
    settings: AppSettings,
    llm: Annotated[LLMProvider, Depends(get_llm)],
    summarizer: Annotated[LLMProvider, Depends(get_summary_llm)],
    today: Today,
    sessions: Annotated[sessionmaker[Session], Depends(get_session_factory)],
) -> StreamingResponse:
    """Same as /ask, as server-sent events: progress stages first, then the answer.

    Events: {"type": "stage", "stage": "planning"|"querying"|"summarizing"|"searching"|
    "reading"}, then {"type": "done", "response": AskResponse} or {"type": "error",
    "status": int, "detail": str}.
    """
    ratelimit.questions.check(user.id, settings.questions_per_minute)
    events: queue.Queue[dict[str, Any] | None] = queue.Queue()
    user_id = user.id

    def work() -> None:
        with sessions() as db:  # its own session: it outlives the request's
            try:
                worker_user = db.get(User, user_id)
                if worker_user is None:
                    raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not authenticated")
                response = run_ask(
                    body, worker_user, db, settings, llm, summarizer, today,
                    on_stage=lambda stage: events.put({"type": "stage", "stage": stage}),
                )  # fmt: skip
                events.put({"type": "done", "response": response.model_dump(mode="json")})
            except HTTPException as exc:
                events.put({"type": "error", "status": exc.status_code, "detail": exc.detail})
            except Exception:
                log.exception("streamed ask failed")
                events.put({"type": "error", "status": 500, "detail": "Something went wrong."})
            finally:
                events.put(None)

    threading.Thread(target=work, daemon=True).start()

    def stream() -> Iterator[str]:
        yield ": start\n\n"  # flush headers through proxies right away
        while (event := events.get()) is not None:
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )


def run_ask(
    body: AskRequest,
    user: User,
    db: Session,
    settings: Settings,
    llm: LLMProvider,
    summarizer: LLMProvider,
    today: date,
    on_stage: Callable[[str], None] | None = None,
) -> AskResponse:
    if body.conversation_id:
        conv = _conversation(db, user, body.conversation_id)
        conv_original = conv.dataset_id
        # A question may switch the chat to another data source (another file or sheet).
        ds = _dataset(db, user, body.dataset_id or conv.dataset_id)
        if ds.id != conv.dataset_id:
            conv.dataset_id = ds.id
    elif body.dataset_id:
        ds = _dataset(db, user, body.dataset_id)
        conv = Conversation(
            workspace_id=user.workspace_id,
            user_id=user.id,
            dataset_id=ds.id,
            title=body.question[:200],
        )
        conv_original = ds.id
    else:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, "Send dataset_id or conversation_id."
        )

    parquet = require_parquet(settings, ds)
    # Only follow-ups asked against the SAME source carry over as context.
    previous = [
        (q, a)
        for q, a in (pairwise(conv.messages) if body.conversation_id else [])
        if (q.dataset_id or conv_original) == ds.id
    ]

    # Requests about the conversation itself, answered directly (no model call).
    if SHEETS_REQUEST.search(body.question):
        return _sheets_reply(db, conv, body, ds, user)
    if CODE_REQUEST.search(body.question) and ds.kind == "table":
        return _code_reply(db, conv, body, ds, previous, user)
    if ds.kind == "table" and is_schedule_request(body.question):
        return _schedule_reply(db, conv, body, previous, user, settings)
    if ds.kind == "table" and is_style_request(body.question, previous):
        return _style_reply(db, conv, body, previous, user, llm, today)
    if ds.kind == "table" and PERCENT_REQUEST.search(body.question):
        return _percent_reply(db, conv, body, previous, user)
    if ds.kind == "table" and CHART_REQUEST.search(body.question):
        return _chart_reply(db, conv, body, ds, previous, user, settings, today)
    fmt = requested_format(body.question)
    if fmt is not None and ds.kind == "table":
        return _file_reply(db, conv, body, previous, user, fmt)
    if ds.kind == "table" and FIX_REQUEST.search(body.question) and _last_doctor(previous):
        return _fix_reply(db, conv, body, ds, user, settings, today)
    if ds.kind == "table" and DOCTOR_REQUEST.search(body.question):
        return _doctor_reply(db, conv, body, ds, user, settings, today)
    if ds.kind == "table" and CHANGES_REQUEST.search(body.question):
        return _changes_reply(db, conv, body, ds, user, settings)
    if EXPORT_REQUEST.search(body.question) and ds.kind != "table":
        return _save_turn(
            db, conv, body,
            content=(
                "Downloads (Excel, PDF, Word, PowerPoint, CSV) work for answers from tables "
                "and spreadsheets. For this document, use the copy button on the answer."
            ),
            status_="answered", plan=None, payload=None, calls=[], user=user,
        )  # fmt: skip

    definition = learning.parse_definition(body.question)
    if definition is not None:
        # "sales means revenue": remember it for this file. No model needed.
        learning.remember_term(
            db, workspace_id=user.workspace_id, dataset_id=ds.id, user_id=user.id, d=definition
        )
        return _save_turn(
            db,
            conv,
            body,
            content=(
                f"Got it. From now on, “{definition.term}” means "
                f"{definition.meaning} for “{ds.name}”."
            ),
            status_="answered",
            plan=None,
            payload=None,
            calls=[],
            user=user,
        )

    content: str
    status_: str
    plan: dict[str, Any] | None
    payload: dict[str, Any] | None
    calls: list[LLMCall]
    try:
        if ds.kind == "document":
            # Earlier questions and answers give follow-ups their context.
            doc_turns = [
                (q.content, a.content)
                for q, a in previous
                if q.role == "user" and a.role == "assistant" and a.status != "failed"
            ]
            doc = answer_document_question(
                llm,
                body.question,
                parquet=parquet,
                history=doc_turns[-2:],
                on_stage=on_stage,
                embedder=get_embedder(settings),
            )
            content, status_, plan, payload, calls = (
                doc.text, doc.status, None, _document_payload(doc), doc.calls
            )  # fmt: skip
        else:
            # Recent answered turns let follow-ups ("now split by city") modify the plan.
            turns: list[tuple[str, dict[str, Any]]] = []
            for q, a in previous:
                if q.role != "user" or a.role != "assistant":
                    continue
                if a.plan:
                    turns.append((q.content, a.plan))
                elif a.status == "clarification":  # so "Yes" means yes to *this*
                    turns.append((q.content, {"clarification_needed": a.content}))
            # What AskData has learned about this file: helpful past answers to similar
            # questions (as examples), and the team's words for things.
            examples = learning.similar_examples(
                db, workspace_id=user.workspace_id, ds=ds, question=body.question, today=today
            )
            remembered = (
                None
                if turns  # follow-ups depend on context: never reuse those
                else learning.remembered_answer(
                    db, workspace_id=user.workspace_id, dataset_id=ds.id, question=body.question
                )
            )
            result = answer_question(
                llm,
                body.question,
                dataset_id=ds.id,
                dataset_name=ds.name,
                dataset_version=ds.version,
                row_count=ds.row_count,
                profile=ds.columns,
                parquet=parquet,
                history=history_messages(examples + turns[-HISTORY_TURNS:]),
                today=today,
                summarizer=summarizer,
                on_stage=on_stage,
                notes=learning.glossary_notes(db, workspace_id=user.workspace_id, dataset_id=ds.id),
                remembered=remembered,
                embedder=get_embedder(settings),
            )
            content, status_, payload, calls = (
                result.summary, result.status, _answer_payload(result, ds), result.calls
            )  # fmt: skip
            plan = result.validated.plan.model_dump(mode="json") if result.validated else None
    except LLMUnavailableError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc

    return _save_turn(
        db, conv, body, content=content, status_=status_, plan=plan, payload=payload,
        calls=calls, user=user,
    )  # fmt: skip


# "How many sheets", "what sheets are there", "tabs in this file"
SHEETS_REQUEST = re.compile(
    r"\b(how\s+many|number\s+of|count\s+of|what|which|list|show|name|all)\b"
    r".{0,24}\b(sheets?|tabs?|worksheets?)\b"
    r"|\b(sheets?|tabs?|worksheets?)\b.{0,16}\b(in|of|are|does|this|the|file|workbook)\b",
    re.I,
)


def _workbook_sheets(db: Session, user: User, ds: Dataset) -> list[Dataset]:
    """All datasets that came from the same uploaded workbook (one per sheet)."""
    if ds.file_format not in ("xlsx", "xls", "ods"):
        return [ds]
    stmt = workspace_scoped(
        select(Dataset).where(
            Dataset.original_filename == ds.original_filename,
            Dataset.file_format == ds.file_format,
            Dataset.created_by_id == ds.created_by_id,
        ),
        Dataset, user,
    ).order_by(Dataset.created_at)  # fmt: skip
    found = list(db.scalars(stmt))
    return found or [ds]


def _sheets_reply(
    db: Session, conv: Conversation, body: AskRequest, ds: Dataset, user: User
) -> AskResponse:
    """Answer questions about the workbook's sheets by listing them (each is a source)."""
    sheets = _workbook_sheets(db, user, ds)
    if ds.file_format not in ("xlsx", "xls", "ods") or len(sheets) <= 1:
        content = (
            f"“{ds.name}” is a single table, not a multi-sheet workbook, so there's "
            "just one sheet with "
            f"{len(ds.columns)} columns and {ds.row_count:,} rows."
        )
        return _save_turn(db, conv, body, content=content, status_="answered", plan=None,
                          payload=None, calls=[], user=user)  # fmt: skip
    rows = [
        [s.sheet_name or s.name, len(s.columns), s.row_count,
         "← this one" if s.id == ds.id else "say “switch” or pick it above"]
        for s in sheets
    ]  # fmt: skip
    payload = _report_payload(
        ["Sheet", "Columns", "Rows", "To ask about it"],
        rows,
        f"The workbook “{ds.original_filename}” has {len(sheets)} sheets.",
    )
    content = (
        f"“{ds.original_filename}” has {len(sheets)} sheets, each available as its "
        "own data source. Switch to one with the “Talking to” picker above, then "
        "ask about it. You're on "
        f"“{ds.sheet_name or ds.name}” now."
    )
    return _save_turn(db, conv, body, content=content, status_="answered", plan=None,
                      payload=payload, calls=[], user=user)  # fmt: skip


CODE_REQUEST = re.compile(
    r"\b(python|pandas|sql|script|jupyter|notebook)\b|\b(the|this|some|write|in|give)\s+code\b",
    re.I,
)
EXPORT_REQUEST = re.compile(r"\b(download|export|save|convert)\b", re.I)
# Which file the user wants. Unmistakable names count on their own; everyday words
# ("slides", "image", "chart") only together with an export-ish verb.
_FORMAT_NAMES = [
    ("pdf", re.compile(r"\bpdf\b", re.I)),
    ("pptx", re.compile(r"\b(power\s?point|pptx?|ppt)\b", re.I)),
    ("docx", re.compile(r"\b(docx?|ms\s+word|word\s+(doc|document|file|report))\b", re.I)),
    ("csv", re.compile(r"\bcsv\b", re.I)),
    ("xlsx", re.compile(r"\b(excel|xlsx|xls)\b", re.I)),
    ("png", re.compile(r"\b(png|jpe?g)\b", re.I)),
]
_FORMAT_WORDS = [
    ("pptx", re.compile(r"\b(slides?|slide\s?deck|deck|presentation)\b", re.I)),
    ("docx", re.compile(r"\b(in|to|as|into)\s+word\b|\bword\b", re.I)),
    ("xlsx", re.compile(r"\b(spreadsheet|workbook|sheet)\b", re.I)),
    ("pdf", re.compile(r"\breport\b", re.I)),
    ("png", re.compile(r"\b(image|picture|photo|chart|graph)\b", re.I)),
]
_EXPORT_VERB = re.compile(
    r"\b(download|export|save|convert|turn|make|create|give|send|get|generate|build|"
    r"as\s+an?|into\s+an?|to\s+an?)\b",
    re.I,
)


def requested_format(question: str) -> str | None:
    """The file format asked for, or None when this isn't a request for a file."""
    for fmt, pattern in _FORMAT_NAMES:
        if pattern.search(question):
            return fmt
    if not _EXPORT_VERB.search(question):
        return None
    for fmt, pattern in _FORMAT_WORDS:
        if pattern.search(question):
            return fmt
    return "xlsx" if EXPORT_REQUEST.search(question) else None


# "What changed since the last upload?", "compare with the previous version"
CHANGES_REQUEST = re.compile(
    r"\bwhat('?s|\s+has|\s+have|\s+is)?\s+(changed|different)\b|\bwhat('?s|\s+is)\s+new\b"
    r"|\b(changes|differences?|diff)\b.{0,30}\b(since|from|between|compared|previous|last|old)"
    r"|\bcompare\b.{0,40}\b(previous|last|old|earlier)\b|\bkya\s+(badla|change)",
    re.I,
)
# "Check my data", "any problems?", "duplicates?", "data quality"
DOCTOR_REQUEST = re.compile(
    r"\bdata\s+(doctor|quality|health|issues?|problems?|errors?|check)\b"
    r"|\bcheck\s+(my|the|this)\s+(data|file|sheet|table)\b"
    r"|\b(any|find)\s+(problems?|issues?|errors?|mistakes?)\b"
    r"|\bclean\s*(up)?\s+(my|the|this)\s+(data|file|sheet|table)\b"
    r"|\b(duplicates?|inconsistent|spelling\s+(mistakes|errors))\b",
    re.I,
)
# "Fix them" after a data doctor report
FIX_REQUEST = re.compile(
    r"^\W*(please\s+|ok\s+|yes\s+)?(fix|clean)(\s+(it|them|all|these|this|those|everything"
    r"|the\s+(data|issues|problems)))?(\s+please)?\W*$",
    re.I,
)


def _schedule_reply(
    db: Session, conv: Conversation, body: AskRequest, previous: list[tuple[Message, Message]],
    user: User, settings: Settings,
) -> AskResponse:  # fmt: skip
    """'Send me this every Monday at 9': schedule the latest computed answer."""
    from app.api.digests import DigestIn, create_digest

    last = next((a for q, a in reversed(previous) if a.role == "assistant" and a.plan
                 and (a.plan or {}).get("metric") != "describe"), None)  # fmt: skip
    if last is None:
        return _save_turn(
            db, conv, body,
            content=("Ask me a question first (for example “open tasks by person”), "
                     "then say “send me this every Monday at 9”."),
            status_="answered", plan=None, payload=None, calls=[], user=user,
        )  # fmt: skip
    s = parse_schedule(body.question)
    digest = create_digest(db, user, DigestIn(message_id=last.id, cadence=s.cadence,
                                              weekday=s.weekday, hour=s.hour, minute=s.minute),
                           settings.default_timezone)  # fmt: skip
    where = ", ".join(digest.recipients)
    content = (f"Scheduled: “{digest.title}” {describe_schedule(digest)}, with fresh "
               f"numbers each time, to {where}. Manage it on the Reports page.")  # fmt: skip
    if settings.resend_api_key is None:
        content += " (Email isn't set up yet, so for now each digest appears on Reports.)"
    return _save_turn(db, conv, body, content=content, status_="answered", plan=None,
                      payload=None, calls=[], user=user)  # fmt: skip


def _last_doctor(previous: list[tuple[Message, Message]]) -> dict[str, Any] | None:
    for _q, a in reversed(previous[-3:]):
        doctor = (a.answer or {}).get("doctor")
        if doctor:
            return dict(doctor)
    return None


def _report_payload(columns: list[str], rows: list[list[Any]], description: str, *,
                    truncated: bool = False, doctor: dict[str, Any] | None = None,
                    ) -> dict[str, Any]:  # fmt: skip
    """A table answer computed by code (no plan): renders and exports like any other."""
    return {
        "table": {"columns": columns, "rows": rows, "truncated": truncated},
        "chart": None,
        "explanation": {"description": description, "sql": "", "params": [],
                        "date_range": None, "plan": {}, "cached": False, "elapsed_ms": 0},
        "summary_source": "template",
        **({"doctor": doctor} if doctor else {}),
    }  # fmt: skip


def _doctor_reply(
    db: Session, conv: Conversation, body: AskRequest, ds: Dataset, user: User,
    settings: Settings, today: date,
) -> AskResponse:  # fmt: skip
    issues, _path = run_doctor(settings, ds, today)
    rows = [[i.severity.capitalize(), i.title, i.column or "", i.rows,
             "; ".join(i.examples[:3]), i.fix_label or ""] for i in issues]  # fmt: skip
    payload = _report_payload(
        ["Severity", "Problem", "Column", "Rows", "Examples", "Automatic fix"], rows,
        f"Data doctor check of \u201c{ds.name}\u201d (version {ds.version}).",
        doctor={"dataset_id": ds.id, "version": ds.version,
                "issues": [i.as_dict() for i in issues]},
    ) if issues else None  # fmt: skip
    return _save_turn(db, conv, body, content=summarize(issues, ds.name), status_="answered",
                      plan=None, payload=payload, calls=[], user=user)  # fmt: skip


def _fix_reply(
    db: Session, conv: Conversation, body: AskRequest, ds: Dataset, user: User,
    settings: Settings, today: date,
) -> AskResponse:  # fmt: skip
    try:
        fixed, changes = fix_dataset(settings, db, user, ds, today, [])
    except HTTPException as exc:
        return _save_turn(db, conv, body, content=str(exc.detail), status_="answered",
                          plan=None, payload=None, calls=[], user=user)  # fmt: skip
    done = "; ".join(i.fix_label or i.title for i in fixed)
    content = (f"Done: {done}. Saved as version {ds.version} of \u201c{ds.name}\u201d "
               f"({changes.rows_before:,} \u2192 {changes.rows_after:,} rows, "
               f"{changes.changed:,} rows changed). The previous version is kept: ask "
               "\u201cwhat changed?\u201d to see every change.")  # fmt: skip
    return _save_turn(db, conv, body, content=content, status_="answered", plan=None,
                      payload=None, calls=[], user=user)  # fmt: skip


def _changes_reply(
    db: Session, conv: Conversation, body: AskRequest, ds: Dataset, user: User,
    settings: Settings,
) -> AskResponse:  # fmt: skip
    storage = _storage(settings)
    previous = storage.ensure_previous(ds.workspace_id, ds.id)
    if previous is None or ds.version < 2:
        return _save_turn(
            db, conv, body,
            content=(f"\u201c{ds.name}\u201d has only one version so far. Upload the newer "
                     "file with \u201cUpload new version\u201d on the Datasets page (for "
                     "example this week's export) and I'll show exactly what changed."),
            status_="answered", plan=None, payload=None, calls=[], user=user,
        )  # fmt: skip
    comparison = compare_versions(previous, require_parquet(settings, ds))
    columns = ["Change", comparison.key or "Row", "Column", "Before", "After"]
    payload = None if comparison.nothing else _report_payload(
        columns, comparison.rows,
        f"Version {ds.version - 1} vs version {ds.version} of \u201c{ds.name}\u201d"
        + (f", rows matched by {comparison.key}." if comparison.key else "."),
        truncated=comparison.truncated,
    )  # fmt: skip
    summary = describe_comparison(comparison, ds.name)
    return _save_turn(db, conv, body, content=summary, status_="answered", plan=None,
                      payload=payload, calls=[], user=user)  # fmt: skip


FORMAT_NAMES = {"xlsx": "Excel workbook", "pdf": "PDF report", "docx": "Word report",
                "pptx": "PowerPoint deck", "csv": "CSV file", "png": "chart image"}  # fmt: skip


# Clearly about how the export looks.
STYLE_REQUEST = re.compile(
    r"\b(highlight|bold|shade)\w*\b"
    r"|\b(header|heading)s?\b.{0,30}\b(colou?r|red|green|blue|yellow|orange|purple|pink|teal"
    r"|gr[ae]y|black|indigo)\b"
    r"|\bcolou?r\s+(the|it|them|rows?|cells?|code)\b"
    r"|\b(add|include|with)\s+(an?\s+|the\s+)?(average|avg|total|sum|min(imum)?|max(imum)?"
    r"|count)\s+rows?\b"
    r"|\b(hide|remove|drop)\s+(the\s+)?[\w ]{1,40}?\s+columns?\b"
    r"|\b(rename|change|set)\s+the\s+title\b",
    re.I,
)
# Only about looks when the last reply was a file ("now sort it by date", "make it green").
STYLE_FOLLOW_UP = re.compile(
    r"\b(sort(ed)?\s+(it\s+|them\s+)?by|red|green|blue|yellow|orange|purple|pink|teal"
    r"|gr[ae]y|stripes?|banded|title)\b",
    re.I,
)


def _last_file(previous: list[tuple[Message, Message]]) -> Message | None:
    for _q, a in reversed(previous):
        if a.role == "assistant" and (a.answer or {}).get("type") == "file":
            return a
        if a.role == "assistant" and (a.answer or {}).get("table") is not None:
            return None  # a newer table answer: a new request is about that one
    return None


def is_style_request(question: str, previous: list[tuple[Message, Message]]) -> bool:
    if not previous:
        return False
    if STYLE_REQUEST.search(question):
        return True
    last = previous[-1][1]
    return (last.answer or {}).get("type") == "file" and bool(STYLE_FOLLOW_UP.search(question))


def _style_reply(
    db: Session,
    conv: Conversation,
    body: AskRequest,
    previous: list[tuple[Message, Message]],
    user: User,
    llm: LLMProvider,
    today: date,
) -> AskResponse:
    """Change how the export looks. The model reads the request; code applies it."""
    last = _last_table_answer(previous)
    if last is None:
        return _save_turn(
            db, conv, body,
            content=(
                "Ask me for the data first (for example \u201copen tasks by due date\u201d), "
                "then tell me how the Excel should look: colours, highlights, extra rows."
            ),
            status_="answered", plan=None, payload=None, calls=[], user=user,
        )  # fmt: skip
    question, answer = last
    file = _last_file(previous)
    current = ExportStyle()
    if file is not None and (file.answer or {}).get("message_id") == answer.id:
        current = ExportStyle.model_validate((file.answer or {}).get("style") or {})
    table = (answer.answer or {})["table"]
    try:
        style, notes, call = interpret_style(
            llm, body.question, columns=table.get("columns", []),
            rows=table.get("rows", [])[:200], current=current, today=today,
        )  # fmt: skip
    except LLMUnavailableError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    fmt = requested_format(body.question)
    if fmt is None and file is not None and (file.answer or {}).get("message_id") == answer.id:
        fmt = (file.answer or {}).get("format")
    fmt = fmt if fmt in FORMAT_NAMES and fmt != "png" else "xlsx"
    content = f"Done. Your {FORMAT_NAMES[fmt]} now has: {describe_style(style)}."
    if notes:
        content += " Note: " + "; ".join(notes) + "."
    return _save_turn(
        db, conv, body, content=content, status_="answered", plan=None,
        payload={"type": "file", "format": fmt, "message_id": answer.id,
                 "label": question.content[:120],
                 "style": style.model_dump(mode="json", exclude_defaults=True)},
        calls=[call], user=user,
    )  # fmt: skip


def _last_table_answer(
    previous: list[tuple[Message, Message]],
) -> tuple[Message, Message] | None:
    for q, a in reversed(previous):
        answer = a.answer or {}
        if q.role == "user" and a.role == "assistant" and "table" in answer:
            return q, a
    return None


# "As percentages", "share of total", "% of total", "what percent of the total"
PERCENT_REQUEST = re.compile(
    r"(percentages?|percent|%|shares?|proportions?).{0,16}\b(of\s+)?(total|whole|all|each)\b"
    r"|\b(as|in|to)\s+(a\s+)?(percentages?|percent|shares?)\b"
    r"|\b(as|in)\s+%|\bshare\s+of\s+total\b",
    re.I,
)


def _percent_reply(
    db: Session, conv: Conversation, body: AskRequest,
    previous: list[tuple[Message, Message]], user: User,
) -> AskResponse:  # fmt: skip
    """Add a '% of total' column to the last grouped answer (computed in code)."""
    last = _last_table_answer(previous)
    if last is None:
        return _save_turn(
            db, conv, body,
            content=("Ask a grouped question first (for example “revenue by region”), "
                     "then say “show as percentages”."),
            status_="answered", plan=None, payload=None, calls=[], user=user,
        )  # fmt: skip
    _question, answer = last
    table = (answer.answer or {}).get("table") or {}
    columns, rows = table.get("columns", []), table.get("rows", [])
    metric = (answer.plan or {}).get("metric")
    values = [r[-1] for r in rows]
    if (
        len(columns) < 2
        or not rows
        or metric not in ("sum", "count", "count_distinct")
        or not all(isinstance(v, int | float) and not isinstance(v, bool) for v in values)
    ):
        return _save_turn(
            db, conv, body,
            content=("Percentages of a total only make sense for a grouped count or sum "
                     "(for example “revenue by region”). This answer isn't one."),
            status_="answered", plan=None, payload=None, calls=[], user=user,
        )  # fmt: skip
    total = sum(float(v) for v in values)
    out_cols = [*columns, "% of total"]
    out_rows = [[*r, round(float(r[-1]) / total * 100, 1) if total else 0.0] for r in rows]
    payload = _report_payload(
        out_cols, out_rows, f"{_pretty(columns[-1])} as a share of the total ({total:,.0f})."
    )
    top = max(rows, key=lambda r: float(r[-1]))
    content = (f"As a share of the total {_pretty(columns[-1]).lower()} "
               f"({total:,.0f}): {top[0]} is the largest at "
               f"{round(float(top[-1]) / total * 100, 1)}%.")  # fmt: skip
    return _save_turn(db, conv, body, content=content, status_="answered", plan=None,
                      payload=payload, calls=[], user=user)  # fmt: skip


# "Make a graph for this", "show that as a bar chart", "plot it", "visualise this"
CHART_REQUEST = re.compile(
    r"\b(bar|line|pie|column)\s+(chart|graph)\b"
    r"|\b(make|show|draw|plot|create|give|turn|convert|visuali[sz]e)\b"
    r".{0,30}\b(chart|graph|plot|visuali[sz])\w*\b"
    r"|\b(chart|graph|plot|visuali[sz]e)\s+(this|that|it|them)\b",
    re.I,
)


def _wanted_chart_type(question: str) -> str:
    q = question.lower()
    if "pie" in q:
        return "pie"
    if "line" in q or "trend" in q or "over time" in q:
        return "line"
    return "bar"


def _chart_reply(
    db: Session, conv: Conversation, body: AskRequest, ds: Dataset,
    previous: list[tuple[Message, Message]], user: User, settings: Settings, today: date,
) -> AskResponse:  # fmt: skip
    """Redraw the last computed answer as a chart (re-runs its plan with a chart type)."""
    last = next(
        (
            (q, a)
            for q, a in reversed(previous)
            if q.role == "user"
            and a.role == "assistant"
            and a.plan
            and (a.plan or {}).get("metric") not in (None, "describe", "rows")
        ),
        None,
    )
    if last is None:
        return _save_turn(
            db, conv, body,
            content=("Ask a question that groups your data first (for example “revenue "
                     "by region” or “tasks by status”), then say “make "
                     "a bar chart”."),
            status_="answered", plan=None, payload=None, calls=[], user=user,
        )  # fmt: skip
    question, answer = last
    plan = {**(answer.plan or {}), "chart": _wanted_chart_type(body.question)}
    try:
        vp = validate_plan(QueryPlan.model_validate(plan), columns_from_profile(ds.columns), today)
        result = execute_plan(
            vp, require_parquet(settings, ds), dataset_id=ds.id, version=ds.version
        )
    except (PlanError, QueryTimeoutError) as exc:
        return _save_turn(db, conv, body, content=str(exc), status_="answered", plan=None,
                          payload=None, calls=[], user=user)  # fmt: skip
    chart = choose_chart(vp, result)
    if chart is None:
        return _save_turn(
            db, conv, body,
            content=("That answer can't be charted — a chart needs groups with at least "
                     "two values (for example “by region” or “by month”). "
                     "Try grouping your question, then ask for a chart."),
            status_="answered", plan=None, payload=None, calls=[], user=user,
        )  # fmt: skip
    a = Answer("answered", answer.content, vp, result, chart, describe_plan(vp), "template")
    content = f"Here's “{question.content}” as a {chart.type} chart."
    return _save_turn(db, conv, body, content=content, status_="answered",
                      plan=vp.plan.model_dump(mode="json"), payload=_answer_payload(a, ds),
                      calls=[], user=user)  # fmt: skip


def _file_reply(
    db: Session,
    conv: Conversation,
    body: AskRequest,
    previous: list[tuple[Message, Message]],
    user: User,
    fmt: str,
) -> AskResponse:
    """A download of the latest table answer in the format asked for (no model call)."""
    last = _last_table_answer(previous)
    if last is None:
        return _save_turn(
            db, conv, body,
            content=(
                "Ask me for the data first (for example \u201ctasks assigned to Sara\u201d), "
                f"then ask for a {FORMAT_NAMES[fmt]} and I'll make one of that answer."
            ),
            status_="answered", plan=None, payload=None, calls=[], user=user,
        )  # fmt: skip
    question, answer = last
    has_chart = bool((answer.answer or {}).get("chart"))
    if fmt == "png" and not has_chart:
        return _save_turn(
            db, conv, body,
            content=(
                "That answer has no chart to save as an image. Ask for it as a chart "
                "(for example \u201cshow that as a bar chart\u201d), then ask for the image."
            ),
            status_="answered", plan=None, payload=None, calls=[], user=user,
        )  # fmt: skip
    file = _last_file(previous)
    kept = None
    if file is not None and (file.answer or {}).get("message_id") == answer.id:
        kept = (file.answer or {}).get("style")  # keep the formatting already chosen
    what = {
        "xlsx": "just the table, with a styled header, filters and number formats",
        "pdf": "just the table, ready to print",
        "docx": "just the table, formatted",
        "pptx": "just the table, a few rows per slide",
        "csv": "the plain data, ready for any tool",
        "png": "the chart as a high-resolution picture",
    }[fmt]  # fmt: skip
    return _save_turn(
        db, conv, body,
        content=f"Here's a {FORMAT_NAMES[fmt]} of \u201c{question.content}\u201d: {what}.",
        status_="answered", plan=None,
        payload={"type": "file", "format": fmt, "message_id": answer.id,
                 "label": question.content[:120], **({"style": kept} if kept else {})},
        calls=[], user=user,
    )  # fmt: skip


def _code_reply(
    db: Session,
    conv: Conversation,
    body: AskRequest,
    ds: Dataset,
    previous: list[tuple[Message, Message]],
    user: User,
) -> AskResponse:
    """Code that reproduces the latest computed answer, generated from its plan."""
    last = next(
        (
            (q, a)
            for q, a in reversed(previous)
            if q.role == "user" and a.role == "assistant" and a.plan and a.answer
        ),
        None,
    )
    if last is None:
        return _save_turn(
            db, conv, body,
            content=(
                "Ask me a question about the data first (for example “top 10 customers "
                "by revenue”), then ask for Python code and I'll give you code that "
                "reproduces that answer."
            ),
            status_="answered", plan=None, payload=None, calls=[], user=user,
        )  # fmt: skip
    question, answer = last
    explanation = (answer.answer or {}).get("explanation", {})
    python = pandas_code(
        answer.plan or {},
        file_name=ds.original_filename,
        file_format=ds.file_format,
        date_range=explanation.get("date_range"),
    )
    blocks = [{"language": "python", "code": python}]
    if explanation.get("sql", "").startswith("SELECT"):
        params = ", ".join(repr(p) for p in explanation.get("params", []))
        blocks.append(
            {
                "language": "sql",
                "code": f"-- DuckDB; parameters in order: {params}\n"
                + explanation["sql"].replace("read_parquet(?)", repr(ds.original_filename)),
            }
        )
    return _save_turn(
        db, conv, body,
        content=(
            f"Here's Python (pandas) code that reproduces “{question.content}”. "
            f"Put {ds.original_filename} next to it, or change the file path."
        ),
        status_="answered", plan=None, payload={"type": "code", "blocks": blocks},
        calls=[], user=user,
    )  # fmt: skip


def _save_turn(
    db: Session,
    conv: Conversation,
    body: AskRequest,
    *,
    content: str,
    status_: str,
    plan: dict[str, Any] | None,
    payload: dict[str, Any] | None,
    calls: list[LLMCall],
    user: User,
) -> AskResponse:
    now = datetime.now(UTC)
    position = len(conv.messages) if body.conversation_id else 0
    question = Message(role="user", content=body.question, position=position,
                       dataset_id=conv.dataset_id, created_at=now)  # fmt: skip
    reply = Message(
        role="assistant",
        content=content,
        status=status_,
        plan=plan,
        answer=payload,
        position=position + 1,
        dataset_id=conv.dataset_id,
        created_at=now,
    )
    conv.messages.extend([question, reply])
    conv.updated_at = now
    db.add(conv)
    for call in calls:
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
    return AskResponse(
        conversation_id=conv.id,
        question=MessageOut.model_validate(question),
        reply=MessageOut.model_validate(reply),
    )


@router.get("/conversations", response_model=list[ConversationSummary])
def list_conversations(user: CurrentUser, db: DbSession) -> list[Conversation]:
    stmt = (
        workspace_scoped(select(Conversation), Conversation, user)
        .where(Conversation.user_id == user.id)
        .order_by(Conversation.updated_at.desc())
    )
    return list(db.scalars(stmt))


@router.get("/conversations/{conversation_id}", response_model=ConversationOut)
def get_conversation(conversation_id: str, user: CurrentUser, db: DbSession) -> Conversation:
    return _conversation(db, user, conversation_id)


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(conversation_id: str, user: CurrentUser, db: DbSession) -> None:
    conv = _conversation(db, user, conversation_id)
    audit.record(db, user, "conversation.deleted", target_type="conversation",
                 target_id=conv.id)  # fmt: skip
    db.delete(conv)
    db.commit()


@router.post("/messages/{message_id}/feedback", response_model=FeedbackOut)
def give_feedback(
    message_id: str, body: FeedbackRequest, user: CurrentUser, db: DbSession
) -> FeedbackOut:
    """👍 teaches AskData this answer was right (it becomes an example for similar
    questions); 👎 makes sure it's never reused or learned from."""
    reply = db.get(Message, message_id)
    if reply is None or reply.role != "assistant":
        raise not_found("Message")
    conv = _conversation(db, user, reply.conversation_id)  # also checks ownership
    question = next((m for m in conv.messages if m.position == reply.position - 1), None)
    reply.rating = body.rating or None
    if question is not None:
        learning.learn_from_rating(db, reply=reply, question=question, conv=conv, user_id=user.id)
    db.commit()
    return FeedbackOut(
        message_id=reply.id,
        rating=reply.rating,
        learned=reply.rating == 1 and reply.plan is not None,
    )


def _export_parts(
    db: DbSession, user: User, message_id: str
) -> tuple[Message, dict[str, Any], str, Dataset | None, ExportStyle | None]:
    reply = db.get(Message, message_id)
    if reply is None or reply.role != "assistant":
        raise not_found("Answer")
    conv = _conversation(db, user, reply.conversation_id)  # checks it's the user's own
    answer = reply.answer or {}
    style = None
    if answer.get("type") == "file":
        style = ExportStyle.model_validate(answer.get("style") or {})
        target = db.get(Message, answer.get("message_id", ""))
        if target is None or target.conversation_id != conv.id:
            raise not_found("Answer")
        reply, answer = target, target.answer or {}
    if answer.get("type", "table") != "table" or "table" not in answer:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "This answer has no table to export.")
    question = next((m for m in conv.messages if m.position == reply.position - 1), None)
    return (
        reply,
        answer,
        question.content if question else "AskData answer",
        db.get(Dataset, reply.dataset_id or conv.dataset_id),
        style,
    )


def _download(content: bytes, media_type: str, question: str, ext: str) -> Response:
    name = re.sub(r"[^A-Za-z0-9]+", "-", question).strip("-")[:60] or "answer"
    return Response(
        content,
        media_type=media_type,
        # no-store: always a fresh file, never an old copy from the browser cache.
        headers={
            "Content-Disposition": f'attachment; filename="{name}.{ext}"',
            "Cache-Control": "no-store",
        },
    )


@router.get("/messages/{message_id}/export.{fmt}")
def export_answer(
    message_id: str, fmt: str, user: CurrentUser, db: DbSession, today: Today
) -> Response:
    """A table answer as Excel, PDF, Word, PowerPoint, CSV or a chart PNG. For a file
    message: the answer it points at, with the formatting asked for in chat."""
    if fmt not in FORMATS:
        raise not_found("Export format")
    builder, media_type, _name = FORMATS[fmt]
    reply, answer, question, ds, style = _export_parts(db, user, message_id)
    try:
        content = builder(
            question=question,
            summary=reply.content,
            table=answer["table"],
            chart=answer.get("chart"),
            dataset_name=ds.name if ds else "",
            created_at=reply.created_at,
            explanation=answer.get("explanation"),
            style=style,
            today=today,
        )
    except NothingToExportError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    audit.record(db, user, "answer.exported", target_type="message", target_id=reply.id,
                 format=fmt)  # fmt: skip
    db.commit()
    # The file is named after what was computed, like its title (not the chat message).
    name = export_title(answer.get("explanation"), ds.name if ds else "")
    return _download(content, media_type, name, fmt)


@router.get("/messages/{message_id}/evidence", response_model=EvidenceOut)
def answer_evidence(
    message_id: str,
    user: CurrentUser,
    db: DbSession,
    settings: AppSettings,
    today: Today,
    group: str | None = None,
) -> EvidenceOut:
    """Proof: the rows behind an answer. `group` (a JSON list of the group's values, as
    in the answer's table) narrows it to one group, e.g. ["North"]."""
    reply = db.get(Message, message_id)
    if reply is None or reply.role != "assistant":
        raise not_found("Answer")
    conv = _conversation(db, user, reply.conversation_id)  # checks it's the user's own
    answer = reply.answer or {}
    explanation = answer.get("explanation") or {}
    if not reply.plan or "table" not in answer:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "This answer has no rows behind it.")
    ds = _dataset(db, user, reply.dataset_id or conv.dataset_id)
    if explanation.get("dataset_version") not in (None, ds.version):
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "The data has a newer version since this answer. Ask the question again to see "
            "its rows.",
        )
    try:
        group_values = json.loads(group) if group else None
        if group_values is not None and not isinstance(group_values, list):
            raise ValueError
    except ValueError:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, "group must be a JSON list."
        ) from None
    try:
        ev = find_evidence(
            plan=reply.plan, explanation=explanation, group=group_values, profile=ds.columns,
            parquet=require_parquet(settings, ds), dataset_id=ds.id, version=ds.version,
            today=today,
        )  # fmt: skip
    except PlanError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    return EvidenceOut(columns=ev.columns, rows=ev.rows, total=ev.total)
