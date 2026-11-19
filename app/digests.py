"""Weekly/daily digests: saved questions re-run on a schedule, shown in the app and
emailed (via Resend) when an email service is set up.

Numbers are recomputed from the current data at every run (the saved plan is
re-validated, so "last week" moves with the calendar). The summary is written by code,
and each number is compared with the previous run's.
"""

import base64
import html
import logging
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.exporting import build_answer_workbook
from app.llm.base import LLMProvider
from app.models import Dataset, Digest, DigestRun, User
from app.query.compare import compare_versions, comparison_counts
from app.query.executor import columns_from_profile, execute_plan, validate_plan
from app.query.presentation import describe_plan, fallback_summary
from app.query.retrieval import apply_retrieval
from app.schemas.query_plan import QueryPlan
from app.storage import get_storage

log = logging.getLogger(__name__)
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
MAX_ITEMS = 10
DIGEST_ROWS = 15


# --- schedules ------------------------------------------------------------------------


def zone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        return ZoneInfo("UTC")


def next_run(cadence: str, weekday: int, hour: int, tz: str, after: datetime) -> datetime:
    """The first scheduled time strictly after `after` (returned in UTC)."""
    local = after.astimezone(zone(tz))
    candidate = local.replace(hour=hour, minute=0, second=0, microsecond=0)
    if cadence == "weekly":
        candidate += timedelta(days=(weekday - candidate.weekday()) % 7)
    while candidate <= local:
        candidate += timedelta(days=7 if cadence == "weekly" else 1)
    return candidate.astimezone(UTC)


@dataclass
class Schedule:
    cadence: str  # daily | weekly
    weekday: int
    hour: int


_TIME = re.compile(r"\b(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)?\b", re.I)
_SEND = re.compile(
    r"\b(send|email|e-mail|mail|schedule|remind|notify|deliver|subscribe|bhejo|bhej\s+do"
    r"|bhejna)\b",
    re.I,
)
_CADENCE = re.compile(
    r"\b(every|each|har)\s+(day|morning|evening|week|weekday|monday|tuesday|wednesday"
    r"|thursday|friday|saturday|sunday|din|roz|hafte|haftay)\b|\b(daily|weekly)\b",
    re.I,
)


def is_schedule_request(question: str) -> bool:
    """'Send me this every Monday at 9', 'email this weekly'. Both a sending word and a
    when-word are needed, so "weekly revenue trend" stays a data question."""
    return bool(_SEND.search(question) and _CADENCE.search(question))


def parse_schedule(text: str) -> Schedule:
    """'every Monday at 9am' / 'daily at 18:00' / 'weekly' -> a schedule (weekly Monday
    9:00 unless said otherwise)."""
    lowered = text.lower()
    weekday = next((i for i, d in enumerate(WEEKDAYS) if d in lowered or d[:3] + " " in lowered),
                   0)  # fmt: skip
    daily = bool(
        re.search(r"\b(daily|every\s+(day|morning)|each\s+day|har\s+(din|roz))\b", lowered)
    )
    hour = 9
    for m in _TIME.finditer(text):
        h = int(m[1])
        suffix = (m[3] or "").replace(".", "").lower()
        if not suffix and not m[2] and not m[0].lower().startswith("at"):
            continue  # a bare number ("top 10") isn't a time
        if suffix == "pm" and h < 12:
            h += 12
        if suffix == "am" and h == 12:
            h = 0
        if 0 <= h <= 23:
            hour = h
            break
    if "evening" in lowered and hour == 9:
        hour = 18
    return Schedule("daily" if daily else "weekly", weekday, hour)


def describe_schedule(d: Digest) -> str:
    when = "every day" if d.cadence == "daily" else f"every {WEEKDAYS[d.weekday].capitalize()}"
    return f"{when} at {d.hour:02d}:00 ({d.timezone})"


# --- running --------------------------------------------------------------------------


def _headline(rows: list[list[Any]]) -> float | None:
    """A single-number answer's value (for 'was X last time')."""
    if len(rows) == 1 and rows[0] and isinstance(rows[0][-1], int | float):
        return float(rows[0][-1])
    return None


def run_digest(
    db: Session, settings: Settings, digest: Digest, *, llm: LLMProvider | None, today: date,
    now: datetime | None = None,
) -> DigestRun:  # fmt: skip
    """Re-run the digest's questions on the current data and save the result."""
    now = now or datetime.now(UTC)
    ds = db.get(Dataset, digest.dataset_id)
    if ds is None:
        raise ValueError("The dataset was deleted.")
    storage = get_storage(settings)
    parquet = storage.ensure_local(ds.workspace_id, ds.id)
    if parquet is None:
        raise ValueError("The dataset's data file is missing.")
    previous_run = db.scalar(select(DigestRun).where(DigestRun.digest_id == digest.id)
                             .order_by(DigestRun.created_at.desc()).limit(1))  # fmt: skip
    before = {i["question"]: i.get("headline") for i in (previous_run.content if previous_run
                                                          else [])}  # fmt: skip

    content: list[dict[str, Any]] = []
    for item in digest.items[:MAX_ITEMS]:
        try:
            vp = validate_plan(QueryPlan.model_validate(item["plan"]),
                               columns_from_profile(ds.columns), today)  # fmt: skip
            if llm is not None and any(f.op == "about" for f in vp.plan.filters):
                vp = apply_retrieval(llm, vp, parquet).plan
            result = execute_plan(vp, parquet, dataset_id=ds.id, version=ds.version)
            headline = _headline(result.rows)
            content.append({
                "question": item["question"], "summary": fallback_summary(vp, result),
                "description": describe_plan(vp), "columns": result.columns,
                "rows": result.rows[:DIGEST_ROWS],
                "truncated": result.truncated or len(result.rows) > DIGEST_ROWS,
                "headline": headline, "previous": before.get(item["question"]),
                "plan": vp.plan.model_dump(mode="json"),
            })  # fmt: skip
        except Exception as exc:  # one broken question shouldn't stop the others
            log.warning("digest %s question failed: %s", digest.id, exc)
            content.append({"question": item["question"], "error": str(exc)[:300],
                            "summary": "This question couldn't be answered with the current "
                                       "data (a column may have been renamed)."})  # fmt: skip

    changes = None
    if previous_run is not None and previous_run.dataset_version != ds.version:
        prev_path = storage.ensure_previous(ds.workspace_id, ds.id)
        if prev_path is not None and ds.version == previous_run.dataset_version + 1:
            changes = comparison_counts(compare_versions(prev_path, parquet))

    run = DigestRun(digest_id=digest.id, workspace_id=digest.workspace_id,
                    dataset_version=ds.version, content=content, changes=changes)  # fmt: skip
    db.add(run)
    db.flush()
    owner = db.get(User, digest.user_id)
    run.emailed, run.email_note = send_digest_email(settings, digest, run, ds, owner)
    digest.last_run_at = now
    digest.next_run_at = next_run(digest.cadence, digest.weekday, digest.hour, digest.timezone,
                                  now)  # fmt: skip
    return run


def run_due_digests(db: Session, settings: Settings, *, llm: LLMProvider | None,
                    now: datetime | None = None) -> list[DigestRun]:  # fmt: skip
    """Send every active digest whose time has come (called by the hourly timer)."""
    now = now or datetime.now(UTC)
    due = list(db.scalars(select(Digest).where(Digest.active.is_(True),
                                               Digest.next_run_at <= now).limit(200)))  # fmt: skip
    runs = []
    for digest in due:
        today = now.astimezone(zone(digest.timezone)).date()
        try:
            runs.append(run_digest(db, settings, digest, llm=llm, today=today, now=now))
        except Exception:
            log.exception("digest %s failed", digest.id)
            digest.next_run_at = next_run(digest.cadence, digest.weekday, digest.hour,
                                          digest.timezone, now)  # fmt: skip
        db.commit()
    return runs


# --- email ----------------------------------------------------------------------------


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:,.2f}" if not v.is_integer() else f"{int(v):,}"
    if isinstance(v, int) and not isinstance(v, bool):
        return f"{v:,}"
    return "" if v is None else str(v)


def _delta(item: dict[str, Any]) -> str:
    now, before = item.get("headline"), item.get("previous")
    if now is None or before is None:
        return ""
    if before == now:
        return "same as last time"
    diff = now - before
    pct = f" ({diff / before:+.0%})" if before else ""
    return f"{'up' if diff > 0 else 'down'} {_fmt(abs(diff))}{pct} since last time"


def render_email(digest: Digest, run: DigestRun, ds: Dataset, app_url: str) -> str:
    e = html.escape
    parts = [
        '<div style="font-family:Segoe UI,Arial,sans-serif;color:#18181b;max-width:680px">',
        f'<div style="height:4px;background:#4f46e5"></div>'
        f'<h2 style="margin:16px 0 4px">{e(digest.title)}</h2>'
        f'<p style="color:#71717a;margin:0 0 16px;font-size:13px">From “{e(ds.name)}” '
        f"(version {run.dataset_version}) · {describe_schedule(digest)}</p>",
    ]
    if run.changes:
        c = run.changes
        parts.append(
            f'<p style="background:#eef2ff;padding:8px 12px;border-radius:6px;font-size:14px">'
            f"New data since the last digest: {c['added']:,} new rows, {c['removed']:,} removed, "
            f"{c['changed']:,} changed.</p>"
        )
    for item in run.content:
        parts.append(f'<h3 style="margin:20px 0 4px;font-size:16px">{e(item["question"])}</h3>')
        delta = _delta(item)
        parts.append(f'<p style="margin:0 0 8px;font-size:14px">{e(item["summary"])}'
                     + (f' <b style="color:#4f46e5">{e(delta)}</b>' if delta else "")
                     + "</p>")  # fmt: skip
        cols, rows = item.get("columns") or [], item.get("rows") or []
        if cols and rows and not (len(rows) == 1 and len(cols) == 1):
            th = "text-align:left;padding:6px 8px;background:#4f46e5;color:#fff;font-size:12px"
            head = "".join(f'<th style="{th}">{e(str(c))}</th>' for c in cols)
            body = "".join(
                "<tr>" + "".join(f'<td style="padding:5px 8px;border-bottom:1px solid #e4e4e7;'
                                 f'font-size:13px">{e(_fmt(v))}</td>' for v in r) + "</tr>"
                for r in rows
            )  # fmt: skip
            parts.append(f'<table style="border-collapse:collapse;width:100%">'
                         f"<tr>{head}</tr>{body}</table>")  # fmt: skip
            if item.get("truncated"):
                parts.append('<p style="color:#71717a;font-size:12px">More rows in the attached '
                             "Excel file.</p>")  # fmt: skip
    if app_url:
        parts.append(f'<p style="margin-top:24px"><a href="{e(app_url)}/reports" '
                     'style="color:#4f46e5">Open in AskData</a></p>')  # fmt: skip
    parts.append('<p style="color:#a1a1aa;font-size:11px;margin-top:24px">Every number is '
                 "computed from your data by AskData, not written by AI.</p></div>")  # fmt: skip
    return "".join(parts)


def send_digest_email(
    settings: Settings, digest: Digest, run: DigestRun, ds: Dataset, owner: User | None,
    *, client: httpx.Client | None = None,
) -> tuple[bool, str | None]:  # fmt: skip
    """(sent, note). Uses Resend (free: 3,000 emails/month) when RESEND_API_KEY is set."""
    recipients = [r for r in digest.recipients if "@" in r][:20]
    if not recipients:
        return False, "No email recipients."
    if settings.resend_api_key is None:
        return False, "Email isn't set up yet (RESEND_API_KEY); the digest is in Reports."
    attachments = []
    for i, item in enumerate(run.content[:5], start=1):
        if item.get("columns") and item.get("rows"):
            data = build_answer_workbook(
                question=item["question"], summary=item["summary"],
                table={"columns": item["columns"], "rows": item["rows"]}, chart=None,
                dataset_name=ds.name, created_at=run.created_at,
                explanation={"description": item.get("description", ""),
                             "plan": item.get("plan") or {}},
            )  # fmt: skip
            name = re.sub(r"[^A-Za-z0-9]+", "-", item["question"]).strip("-")[:50] or f"item-{i}"
            attachments.append({"filename": f"{name}.xlsx",
                                "content": base64.b64encode(data).decode()})  # fmt: skip
    payload = {
        "from": settings.digest_from_email,
        "to": recipients,
        "subject": f"{digest.title} — {run.created_at:%d %b %Y}",
        "html": render_email(digest, run, ds, settings.app_url.rstrip("/")),
        "attachments": attachments,
    }
    own = client is None
    http = client or httpx.Client(timeout=30)
    try:
        resp = http.post("https://api.resend.com/emails", json=payload, headers={
            "Authorization": f"Bearer {settings.resend_api_key.get_secret_value()}"})  # fmt: skip
        if resp.status_code >= 400:
            detail = resp.text[:200]
            log.warning("digest email failed (%s): %s", resp.status_code, detail)
            return False, f"The email service refused it ({resp.status_code}): {detail}"
        return True, None
    except httpx.HTTPError as exc:
        return False, f"Couldn't reach the email service: {exc}"
    finally:
        if own:
            http.close()
