"""The full ask flow: plan -> execute -> chart -> summary. No database writes here."""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from app.llm.base import ChatMessage, LLMCall, LLMProvider, LLMUnavailableError
from app.query.executor import (
    PlanError,
    QueryResult,
    QueryTimeoutError,
    ValidatedPlan,
    columns_from_profile,
    execute_plan,
    validate_plan,
)
from app.query.planner import PlanOutcome, make_plan
from app.query.presentation import (
    ChartSpec,
    choose_chart,
    describe_plan,
    fallback_summary,
    fmt_value,
    summary_is_grounded,
)
from app.query.retrieval import apply_retrieval, value_hints

if TYPE_CHECKING:
    from app.learning import Remembered

SUMMARY_ROWS = 25

Status = Literal["answered", "clarification", "failed"]


@dataclass
class Answer:
    status: Status
    summary: str
    validated: ValidatedPlan | None = None
    result: QueryResult | None = None
    chart: ChartSpec | None = None
    description: str | None = None
    summary_source: Literal["llm", "template", "none"] = "none"
    calls: list[LLMCall] = field(default_factory=list)
    reused: bool = False  # answered from an identical earlier question, without the model
    # Table RAG: what retrieval did ("Matched 'zarmeen' to 'Zarmeen Khan'"), and the
    # rows each topic filter matched (filter index -> row numbers), kept for proof.
    retrieval_notes: list[str] = field(default_factory=list)
    topic_rows: dict[int, list[int]] = field(default_factory=dict)


def describe_profile(profile: list[dict[str, Any]], row_count: int) -> QueryResult:
    """'What is this data about?' answered from the stored column summary (no SQL)."""
    rows: list[list[Any]] = []
    for c in profile:
        if c["kind"] in ("number", "date", "datetime") and c.get("min") is not None:
            detail = f"{fmt_value(c['min'])} to {fmt_value(c['max'])}"
        else:
            detail = ", ".join(fmt_value(v) for v in c.get("samples", []))
        rows.append([c["name"], c["kind"], detail])
    return QueryResult(
        columns=["column", "type", "examples or range"],
        rows=rows,
        truncated=False,
        sql=f"-- no query: described from the column summary ({row_count:,} rows)",
        params=[],
        dimensions=[],
        value_column="",
    )


def table_text(result: QueryResult, limit: int = SUMMARY_ROWS) -> str:
    lines = [" | ".join(result.columns)]
    for row in result.rows[:limit]:
        lines.append(" | ".join(fmt_value(v) if v is not None else "(empty)" for v in row))
    shown = min(len(result.rows), limit)
    note = f"({shown} of {len(result.rows)} rows shown"
    note += "; more groups exist beyond the limit)" if result.truncated else ")"
    facts = key_facts(result)
    text = "\n".join(lines) + "\n" + note
    return text + ("\n\nKey facts (computed):\n" + "\n".join(facts) if facts else "")


def key_facts(result: QueryResult) -> list[str]:
    """Highest/lowest (and first/last for trends), so small models needn't scan rows."""
    rows = [
        r for r in result.rows if isinstance(r[-1], int | float) and not isinstance(r[-1], bool)
    ]
    if not result.dimensions or len(rows) < 2:
        return []

    def label(r: list[Any]) -> str:
        return " / ".join(fmt_value(v) for v in r[:-1])

    hi = max(rows, key=lambda r: r[-1])
    lo = min(rows, key=lambda r: r[-1])
    facts = [
        f"- Highest: {label(hi)} with {fmt_value(hi[-1])}",
        f"- Lowest: {label(lo)} with {fmt_value(lo[-1])}",
    ]
    if result.truncated:
        facts[1] += " (lowest of the rows shown only)"
    return facts


def _summarize(
    llm: LLMProvider, question: str, vp: ValidatedPlan, result: QueryResult, description: str
) -> tuple[str, Literal["llm", "template"], list[LLMCall]]:
    if not result.rows:
        return fallback_summary(vp, result), "template", []
    try:
        text, call = llm.summarize(question, description, table_text(result))
    except LLMUnavailableError:
        return fallback_summary(vp, result), "template", []
    # Numbers the summary may mention besides the table (never topic row numbers).
    shown = [v for f, v in zip(vp.plan.filters, vp.filter_values, strict=True) if f.op != "about"]
    extra: list[Any] = [vp.plan.limit, *shown]
    if vp.date_range:
        extra += [vp.date_range.start.isoformat(), vp.date_range.last_day.isoformat()]
    if text and summary_is_grounded(text, result, question, extra):
        return text, "llm", [call]
    # The model mentioned a number we didn't compute: never show it.
    return fallback_summary(vp, result), "template", [call]


def answer_question(
    llm: LLMProvider,
    question: str,
    *,
    dataset_id: str,
    dataset_name: str,
    dataset_version: int,
    row_count: int,
    profile: list[dict[str, Any]],
    parquet: Path,
    history: list[ChatMessage],
    today: date,
    summarizer: LLMProvider | None = None,
    on_stage: Callable[[str], None] | None = None,
    notes: str = "",
    remembered: "Remembered | None" = None,
) -> Answer:
    """Raises LLMUnavailableError if the model can't be reached during planning."""
    stage = on_stage or (lambda _s: None)
    stage("planning")
    reused = None
    if remembered is not None:  # same opening question asked before: skip the model
        try:
            reused = validate_plan(remembered.plan, columns_from_profile(profile), today)
        except PlanError:
            reused = None
    if reused is not None:
        outcome = PlanOutcome(validated=reused)
    else:
        try:  # real values that match words in the question, for the planner
            hints = value_hints(question, parquet, profile)
        except Exception:  # retrieval must never stop an answer
            hints = ""
        notes = "\n".join(n for n in (notes, hints) if n)
        outcome = make_plan(
            llm,
            question,
            dataset_name=dataset_name,
            row_count=row_count,
            profile=profile,
            history=history,
            today=today,
            notes=notes,
        )
    if outcome.clarification:
        return Answer("clarification", outcome.clarification, calls=outcome.calls)
    if outcome.validated is None:
        return Answer("failed", outcome.failure or "Please rephrase.", calls=outcome.calls)

    vp = outcome.validated
    retrieval_notes: list[str] = []
    topic_rows: dict[int, list[int]] = {}
    if vp.plan.metric != "describe" and vp.plan.filters:
        if any(f.op == "about" for f in vp.plan.filters):
            stage("matching")
        try:
            found = apply_retrieval(llm, vp, parquet)
        except LLMUnavailableError:
            raise
        except PlanError as exc:
            return Answer("failed", str(exc), vp, calls=outcome.calls)
        vp, retrieval_notes, topic_rows = found.plan, found.notes, found.topic_rows
        outcome.calls += found.calls
    stage("querying")
    try:
        if vp.plan.metric == "describe":
            result = describe_profile(profile, row_count)
        else:
            result = execute_plan(vp, parquet, dataset_id=dataset_id, version=dataset_version)
    except QueryTimeoutError as exc:
        return Answer("failed", f"{exc} Try a narrower question.", vp, calls=outcome.calls)

    description = describe_plan(vp)
    stage("summarizing")
    source: Literal["llm", "template"]
    calls: list[LLMCall]
    if remembered is not None and reused is not None and remembered.rows == result.rows:
        # Same plan, same numbers: the earlier summary still holds. No model call at all.
        source = "llm" if remembered.summary_source == "llm" else "template"
        summary, calls = remembered.summary, []
    else:
        summary, source, calls = _summarize(summarizer or llm, question, vp, result, description)
    return Answer(
        "answered",
        summary,
        vp,
        result,
        choose_chart(vp, result),
        description,
        source,
        outcome.calls + calls,
        reused=reused is not None,
        retrieval_notes=retrieval_notes,
        topic_rows=topic_rows,
    )
