"""Everything shown next to a result that code (not the LLM) is responsible for:
the plain-English description of the plan, the chart choice, a template summary,
and the guard that rejects LLM summaries containing numbers not in the result.
"""

import math
import re
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Literal

from app.query.executor import QueryResult, ValidatedPlan

ChartKind = Literal["bar", "line", "pie"]

_METRIC_WORDS = {
    "count": "Number of rows",
    "count_distinct": "Number of distinct",
    "sum": "Total",
    "avg": "Average",
    "min": "Minimum",
    "max": "Maximum",
}
_OP_WORDS = {
    "=": "is",
    "!=": "is not",
    ">": "is greater than",
    ">=": "is at least",
    "<": "is less than",
    "<=": "is at most",
    "in": "is one of",
    "contains": "contains",
    "between": "is between",
    "about": "is about",
}


def fmt_value(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.2f}".rstrip("0").rstrip(".") if value % 1 else f"{value:,.0f}"
    if isinstance(value, list):
        return ", ".join(fmt_value(v) for v in value)
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def describe_plan(vp: ValidatedPlan) -> str:
    """A plain-English sentence describing exactly what was computed."""
    p = vp.plan
    if p.metric == "describe":
        return "An overview of the dataset: its columns, their types and example values."
    if p.metric == "rows":
        what = "Rows" + (f" showing {', '.join(p.columns)}" if p.columns else "")
    elif p.metric == "count":
        what = "Number of rows"
    elif p.metric == "count_distinct":
        what = f"Number of distinct {p.metric_column}"
    else:
        what = f"{_METRIC_WORDS[p.metric]} {p.metric_column}"

    parts = [what]
    groups = ([f"{p.time_grain} of {p.date_column}"] if p.time_grain else []) + p.group_by
    if groups:
        parts.append("by " + " and ".join(groups))
    conditions = []
    if vp.date_range and p.date_column:
        dr = vp.date_range
        conditions.append(
            f"{p.date_column} from {dr.start.isoformat()} to {dr.last_day.isoformat()} ({dr.label})"
        )
    for f, v in zip(p.filters, vp.filter_values, strict=True):
        if v is None:
            conditions.append(f"{f.column} is {'empty' if f.op == '=' else 'not empty'}")
        elif f.op == "between":
            conditions.append(f"{f.column} is between {fmt_value(v[0])} and {fmt_value(v[1])}")
        elif f.op == "about":  # v is the matched row numbers; describe the topic
            conditions.append(f"{f.column} is about “{f.value}”")
        else:
            conditions.append(f"{f.column} {_OP_WORDS[f.op]} {fmt_value(v)}")
    if conditions:
        parts.append("where " + " and ".join(conditions))
    sentence = ", ".join(parts)
    if p.metric == "rows":
        if p.sort and p.sort.by != "value":
            order = "highest first" if p.sort.direction == "desc" else "lowest first"
            sentence += f"; sorted by {p.sort.by}, {order}"
        return sentence + f"; showing up to {p.limit} rows."
    if groups:
        if p.sort:
            key = "value" if p.sort.by == "value" else p.sort.by
            order = "highest first" if p.sort.direction == "desc" else "lowest first"
            sentence += f"; sorted by {key}, {order}"
        sentence += f"; showing up to {p.limit} rows"
    return sentence + "."


# --- charts ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChartSpec:
    type: ChartKind
    x: str
    y: str
    series: str | None = None


def choose_chart(vp: ValidatedPlan, result: QueryResult) -> ChartSpec | None:
    """Use the requested chart if it fits the data; otherwise pick a sensible default."""
    dims, y, rows = result.dimensions, result.value_column, result.rows
    if not dims or len(rows) < 2:
        return None
    numeric = [
        r[-1] for r in rows if isinstance(r[-1], int | float) and not isinstance(r[-1], bool)
    ]
    if len(numeric) < 2:
        return None  # nothing to plot (e.g. every value is empty): no blank chart
    x = dims[0]
    series = dims[1] if len(dims) > 1 else None
    is_time = vp.plan.time_grain is not None

    def fits(kind: str) -> bool:
        if kind == "line":
            return is_time
        if kind == "bar":
            return len(rows) <= 60
        if kind == "pie":
            values = [r[-1] for r in rows]
            return (
                series is None
                and not is_time
                and len(rows) <= 8
                and vp.plan.metric in ("count", "sum", "count_distinct")
                and all(isinstance(v, int | float) and v >= 0 for v in values)
            )
        return False

    requested = vp.plan.chart
    if requested != "none" and fits(requested):
        return ChartSpec(requested, x, y, series)
    # Auto-chart: even when none was requested, draw one if the data suits it
    # (a line for a time trend, otherwise a bar). Single values stay chart-free.
    for kind in ("line", "bar"):
        if fits(kind):
            return ChartSpec(kind, x, y, series)
    return None


# --- summaries ------------------------------------------------------------------------


def metric_label(vp: ValidatedPlan) -> str:
    """'total revenue', 'average annual_salary', 'number of distinct customer', 'count'."""
    p = vp.plan
    if p.metric == "count":
        return "count"
    if p.metric == "count_distinct":
        return f"number of distinct {p.metric_column}"
    return f"{_METRIC_WORDS[p.metric].lower()} {p.metric_column}"


_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def period_label(value: Any, grain: str | None) -> str:
    """'2025-01-01' -> 'Jan 2025' (month), 'Q1 2025' (quarter), '2025' (year)."""
    text = fmt_value(value)
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", text)
    if not m or grain not in ("month", "quarter", "year"):
        return text
    year, month = m[1], int(m[2])
    if grain == "year":
        return year
    if grain == "quarter":
        return f"Q{(month - 1) // 3 + 1} {year}"
    return f"{_MONTHS[month - 1]} {year}"


def fallback_summary(vp: ValidatedPlan, result: QueryResult) -> str:
    """A deterministic summary built only from the computed rows."""
    if not result.rows:
        return "No rows matched this question. Try widening the time period or removing a filter."
    if vp.plan.metric == "describe":
        names = ", ".join(str(r[0]) for r in result.rows[:12])
        more = " and more" if len(result.rows) > 12 else ""
        return f"This dataset has {len(result.rows)} columns: {names}{more}."
    if vp.plan.metric == "rows":
        more = " (more exist; add a filter or ask for more)" if result.truncated else ""
        return f"Here are the {len(result.rows)} matching rows{more}."
    what = metric_label(vp)
    if not result.dimensions:
        value = fmt_value(result.rows[0][-1])
        if vp.plan.metric == "count":
            return f"There are {value} matching rows."
        return f"The {what} is {value}."

    rows = [
        r for r in result.rows if isinstance(r[-1], int | float) and not isinstance(r[-1], bool)
    ]
    grain = vp.plan.time_grain

    def label(r: list[Any]) -> str:
        parts = (
            [period_label(r[0], grain)] + [fmt_value(v) for v in r[1:-1]]
            if grain
            else [fmt_value(v) for v in r[:-1]]
        )
        return " / ".join(parts)

    if len(rows) < 2:
        only = result.rows[0]
        return f"{label(only)}: {what} of {fmt_value(only[-1])}."
    hi = max(rows, key=lambda r: r[-1])
    lo = min(rows, key=lambda r: r[-1])
    noun = "periods" if grain else "groups"
    sentence = (
        f"{what.capitalize()} was highest for {label(hi)} ({fmt_value(hi[-1])}) and lowest for "
        f"{label(lo)} ({fmt_value(lo[-1])}), across {len(result.rows)} {noun}."
    )
    if result.truncated:
        sentence += " More groups exist beyond the ones shown."
    return sentence


_NUM = re.compile(
    r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?\s*(?:%|k\b|m\b|bn\b|million\b|billion\b|thousand\b)?", re.I
)
_ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_SCALE = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "bn": 1e9, "k": 1e3, "m": 1e6}


def _numbers_in(text: str) -> list[tuple[float, bool]]:
    """Numbers in `text` as (value, was_percent), with k/million/billion applied."""
    out = []
    for m in _NUM.finditer(text):
        token = m.group(0).strip().lower()
        percent = token.endswith("%")
        token = token.rstrip("%")
        scale = 1.0
        for suffix, mult in _SCALE.items():
            if token.endswith(suffix):
                token, scale = token[: -len(suffix)].strip(), mult
                break
        try:
            out.append((float(token.replace(",", "")) * scale, percent))
        except ValueError:
            continue
    return out


def _round_sig(x: float, sig: int) -> float:
    """Round half-up (as people do), not Python's banker's rounding: 41,250 -> 41,300."""
    if x == 0 or not math.isfinite(x):
        return x
    exponent = math.floor(math.log10(abs(x))) - sig + 1
    quantum = Decimal(1).scaleb(exponent)
    return float(Decimal(repr(x)).quantize(quantum, rounding=ROUND_HALF_UP))


def _is_rounding_of(n: float, a: float) -> bool:
    """True if n is a (2+ significant figure) rounding of a, e.g. 1.2e6 of 1,234,567."""
    if math.isclose(n, a, rel_tol=1e-9, abs_tol=1e-9):
        return True
    return any(math.isclose(n, _round_sig(a, sig), rel_tol=1e-9) for sig in range(2, 16))


def _allowed_numbers(
    result: QueryResult, question: str, extra: list[Any]
) -> tuple[list[float], set[str]]:
    nums: list[float] = [float(len(result.rows))]
    dates: set[str] = set()

    def add(v: Any) -> None:
        if isinstance(v, bool) or v is None:
            return
        if isinstance(v, int | float):
            nums.append(float(v))
        elif isinstance(v, str):
            for d in _ISO_DATE.findall(v):
                dates.add(d)
                nums.extend(float(part) for part in d.split("-"))
            nums.extend(n for n, _ in _numbers_in(_ISO_DATE.sub(" ", v)))

    for row in result.rows:
        for v in row:
            add(v)
    for v in extra:
        add(v)
    nums.extend(n for n, _ in _numbers_in(question))
    nums.extend(float(i) for i in range(11))  # "two regions", "top 3", "0 rows"
    return nums, dates


def summary_is_grounded(summary: str, result: QueryResult, question: str, extra: list[Any]) -> bool:
    """True if every number and date in `summary` is backed by the result (or the question)."""
    allowed, allowed_dates = _allowed_numbers(result, question, extra)
    for d in _ISO_DATE.findall(summary):
        if d not in allowed_dates:
            return False
    for n, percent in _numbers_in(_ISO_DATE.sub(" ", summary)):
        candidates = [n, n / 100] if percent else [n]
        if not any(_is_rounding_of(c, a) for c in candidates for a in allowed):
            return False
    return True


def text_is_grounded(answer: str, sources: list[str], question: str) -> bool:
    """Document answers: every number and date must come from the cited passages."""
    # Citation markers like [1] aren't claims. Every other number must be in the sources:
    # no free small numbers here, since "6 months" vs "3 months" is exactly the kind of
    # mistake that matters in a document.
    answer = re.sub(r"\[\d+\]", " ", answer)
    allowed = [n for text in [*sources, question] for n, _ in _numbers_in(text)]
    allowed_dates = {d for text in sources for d in _ISO_DATE.findall(text)}
    for d in _ISO_DATE.findall(answer):
        if d not in allowed_dates:
            return False
    for n, percent in _numbers_in(_ISO_DATE.sub(" ", answer)):
        candidates = [n, n / 100, n * 100] if percent else [n]
        if not any(_is_rounding_of(c, a) for c in candidates for a in allowed):
            return False
    return True
