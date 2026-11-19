"""Question -> validated plan, with one retry that tells the model what was wrong."""

import json
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from app.llm.base import ChatMessage, LLMCall, LLMProvider
from app.llm.prompts import compact_plan
from app.query.executor import PlanError, ValidatedPlan, columns_from_profile, validate_plan
from app.schemas.query_plan import QueryPlan

MAX_ATTEMPTS = 2  # first try + one retry with the validation error
REPHRASE = (
    "I couldn't turn that into a query for this dataset. Could you rephrase it? "
    "Mentioning the column you mean (for example a name from the column list) helps."
)


@dataclass
class PlanOutcome:
    validated: ValidatedPlan | None = None
    clarification: str | None = None
    failure: str | None = None
    calls: list[LLMCall] = field(default_factory=list)


def _fmt_num(v: Any) -> str:
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return f"{v:,}" if isinstance(v, int) else str(v)


def schema_text(name: str, row_count: int, profile: list[dict[str, Any]]) -> str:
    """What the model sees about the data: names, kinds, a few samples, ranges. Never rows."""
    lines = [f'Table "{name}" ({row_count:,} rows). Columns:']
    for c in profile:
        kind = c["kind"]
        if kind in ("number", "date", "datetime") and c.get("min") is not None:
            detail = f"{kind}; {_fmt_num(c['min'])} to {_fmt_num(c['max'])}"
        elif kind == "text":
            samples = ", ".join(json.dumps(s, ensure_ascii=False) for s in c.get("samples", []))
            detail = f"text; e.g. {samples}; {c.get('distinct_count', 0):,} distinct"
        else:
            detail = kind
        lines.append(f"- {c['name']} ({detail})")
    return "\n".join(lines)


def history_messages(turns: list[tuple[str, dict[str, Any]]]) -> list[ChatMessage]:
    """Previous turns as (question, plan), so follow-ups can modify the last plan.

    A clarifying question the model asked is stored as {"clarification_needed": ...}, so
    a reply like "Yes" or "the formulas" is read together with what was asked.
    """
    msgs: list[ChatMessage] = []
    for question, plan in turns:
        msgs.append(ChatMessage("user", f"Question: {question}"))
        msgs.append(ChatMessage("assistant", compact_plan(plan)))
    return msgs


def make_plan(
    llm: LLMProvider,
    question: str,
    *,
    dataset_name: str,
    row_count: int,
    profile: list[dict[str, Any]],
    history: list[ChatMessage],
    today: date,
) -> PlanOutcome:
    schema = schema_text(dataset_name, row_count, profile)
    columns = columns_from_profile(profile)
    outcome = PlanOutcome()
    feedback: tuple[str, str] | None = None
    last_user_facing: str | None = None

    for _ in range(MAX_ATTEMPTS):
        attempt = llm.plan(question, schema, history, feedback)
        outcome.calls.append(attempt.call)
        plan: QueryPlan | None = attempt.plan
        if plan is None:
            problem = attempt.error or "the reply was not a valid JSON plan"
        elif plan.clarification_needed:
            outcome.clarification = plan.clarification_needed.strip()
            return outcome
        else:
            try:
                outcome.validated = validate_plan(plan, columns, today)
                return outcome
            except PlanError as exc:
                problem = last_user_facing = str(exc)
        feedback = (attempt.raw, problem)

    outcome.failure = REPHRASE + (f"\n\nDetails: {last_user_facing}" if last_user_facing else "")
    return outcome
