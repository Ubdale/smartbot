"""Proof: the exact rows behind an answer, or behind one group of it ("North: 38").

Re-runs the answer's own plan (same filters, same period, same rows a topic filter
matched) as a list of rows, plus one more filter per group value that was clicked.
"""

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from app.query.dates import DateRange
from app.query.executor import (
    PlanError,
    ValidatedPlan,
    columns_from_profile,
    execute_plan,
    validate_plan,
)
from app.schemas.query_plan import Filter, QueryPlan

MAX_EVIDENCE_ROWS = 500


@dataclass
class Evidence:
    columns: list[str]
    rows: list[list[Any]]
    total: int  # all matching rows (rows shows at most MAX_EVIDENCE_ROWS)


def _next_period(start: date, grain: str) -> date:
    if grain == "day":
        return start + timedelta(days=1)
    if grain == "week":
        return start + timedelta(days=7)
    months = {"month": 1, "quarter": 3, "year": 12}[grain]
    m = start.month - 1 + months
    return date(start.year + m // 12, m % 12 + 1, 1)


def find_evidence(
    *,
    plan: dict[str, Any],
    explanation: dict[str, Any],
    group: list[Any] | None,
    profile: list[dict[str, Any]],
    parquet: Path,
    dataset_id: str,
    version: int,
    today: date,
) -> Evidence:
    """Raises PlanError when the answer has no rows to show (e.g. "what is this about")."""
    original = QueryPlan.model_validate(plan)
    if original.metric == "describe":
        raise PlanError("This answer describes the file; it isn't computed from rows.")
    vp = validate_plan(original, columns_from_profile(profile), today)

    # The same period as the original answer, even if "last month" has moved on since.
    dr = explanation.get("date_range")
    date_range = (
        DateRange(date.fromisoformat(dr["start"]),
                  date.fromisoformat(dr["end"]) + timedelta(days=1), dr["label"])
        if dr else vp.date_range
    )  # fmt: skip

    filters = list(vp.plan.filters)
    values = list(vp.filter_values)
    topic_rows = explanation.get("topic_rows") or {}
    for i, f in enumerate(filters):
        if f.op == "about":
            values[i] = [int(r) for r in topic_rows.get(str(i), [])]

    # One filter per clicked group value: time period first, then the group columns.
    dims = vp.dimension_aliases
    if group is not None:
        if len(group) != len(dims):
            raise PlanError("That group doesn't belong to this answer.")
        for name, value in zip(dims, group, strict=True):
            if name == vp.plan.time_grain and vp.plan.date_column:
                start = date.fromisoformat(str(value)[:10])
                end = _next_period(start, name)
                dc = vp.plan.date_column
                filters += [Filter(column=dc, op=">=", value=start.isoformat()),
                            Filter(column=dc, op="<", value=end.isoformat())]  # fmt: skip
                values += [start, end]
            else:
                filters.append(Filter(column=name, op="=", value=value))
                values.append(value)

    kinds = dict(vp.kinds)
    for c in columns_from_profile(profile):
        kinds.setdefault(c.name, c.kind)

    def run(metric: str, limit: int) -> Any:
        p = vp.plan.model_copy(update={
            "metric": metric, "metric_column": None, "columns": [], "group_by": [],
            "time_grain": None, "filters": filters, "sort": None, "limit": limit,
            "chart": "none",
        })  # fmt: skip
        return execute_plan(ValidatedPlan(p, kinds, date_range, values), parquet,
                            dataset_id=dataset_id, version=version)  # fmt: skip

    listed = run("rows", MAX_EVIDENCE_ROWS)
    counted = run("count", 1)
    total = int(counted.rows[0][0]) if counted.rows else len(listed.rows)
    return Evidence(listed.columns, listed.rows, total)
