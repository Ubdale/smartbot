"""The full ask flow: plan -> execute -> chart -> summary. No database writes here."""

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Literal

from app.llm.base import ChatMessage, LLMCall, LLMProvider, LLMUnavailableError
from app.query.executor import QueryResult, QueryTimeoutError, ValidatedPlan, execute_plan
from app.query.planner import make_plan
from app.query.presentation import (
    ChartSpec,
    choose_chart,
    describe_plan,
    fallback_summary,
    fmt_value,
    summary_is_grounded,
)

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
    extra: list[Any] = [vp.plan.limit, *vp.filter_values]
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
) -> Answer:
    """Raises LLMUnavailableError if the model can't be reached during planning."""
    outcome = make_plan(
        llm,
        question,
        dataset_name=dataset_name,
        row_count=row_count,
        profile=profile,
        history=history,
        today=today,
    )
    if outcome.clarification:
        return Answer("clarification", outcome.clarification, calls=outcome.calls)
    if outcome.validated is None:
        return Answer("failed", outcome.failure or "Please rephrase.", calls=outcome.calls)

    vp = outcome.validated
    try:
        if vp.plan.metric == "describe":
            result = describe_profile(profile, row_count)
        else:
            result = execute_plan(vp, parquet, dataset_id=dataset_id, version=dataset_version)
    except QueryTimeoutError as exc:
        return Answer("failed", f"{exc} Try a narrower question.", vp, calls=outcome.calls)

    description = describe_plan(vp)
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
    )
