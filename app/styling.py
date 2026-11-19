"""Style exports by asking: "make the header green, highlight overdue tasks in red,
add an average row, sort by due date".

The model only turns the request into an ExportStyle (JSON). Code checks every column,
maps colour names, decides which rows a highlight rule matches (dates relative to
today are resolved here), and computes summary rows. Styles build on each other.
"""

import json
import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from functools import cache
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.llm.base import ChatMessage, LLMCall, LLMProvider, LLMUnavailableError

HighlightOp = Literal[
    "=", "!=", ">", ">=", "<", "<=", "contains", "empty", "not_empty",
    "before_today", "after_today", "within_next_days",
]  # fmt: skip
SummaryKind = Literal["total", "average", "min", "max", "count"]


class Highlight(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: str
    op: HighlightOp
    value: str | float | None = None  # e.g. "Done", 1000, or a number of days
    color: str = "red"


class SortSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: str
    descending: bool = False


class ExportStyle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = None
    header_color: str | None = None
    highlights: list[Highlight] = Field(default_factory=list, max_length=10)
    summary_rows: list[SummaryKind] | None = None  # None = the default (totals when sensible)
    sort: SortSpec | None = None
    hide_columns: list[str] = Field(default_factory=list, max_length=50)
    banded_rows: bool = True


# Strong colours for headers, soft fills (+ readable text) for highlighted rows.
COLORS: dict[str, tuple[str, str, str]] = {  # name: (strong, soft fill, soft text)
    "indigo": ("4F46E5", "E0E7FF", "3730A3"),
    "blue": ("2563EB", "DBEAFE", "1E40AF"),
    "green": ("16A34A", "DCFCE7", "166534"),
    "red": ("DC2626", "FEE2E2", "991B1B"),
    "orange": ("EA580C", "FFEDD5", "9A3412"),
    "yellow": ("CA8A04", "FEF9C3", "854D0E"),
    "purple": ("9333EA", "F3E8FF", "6B21A8"),
    "pink": ("DB2777", "FCE7F3", "9D174D"),
    "teal": ("0D9488", "CCFBF1", "115E59"),
    "gray": ("52525B", "F4F4F5", "3F3F46"),
    "black": ("18181B", "E4E4E7", "18181B"),
}
_ALIASES = {"grey": "gray", "violet": "purple", "amber": "yellow", "navy": "blue", "cyan": "teal"}
_HEX = re.compile(r"^#?([0-9a-fA-F]{6})$")


def color(name: str | None, *, strong: bool) -> tuple[str, str] | None:
    """(fill, text) hex for a colour name or #RRGGBB."""
    if not name:
        return None
    key = name.strip().lower()
    if m := _HEX.match(key):
        return (m[1].upper(), "FFFFFF" if strong else "18181B")
    key = _ALIASES.get(key, key)
    if key not in COLORS:
        return None
    s, fill, text = COLORS[key]
    return (s, "FFFFFF") if strong else (fill, text)


# --- understanding the request (model) ------------------------------------------------

STYLE_SYSTEM = """You turn a request about how a spreadsheet should LOOK into JSON settings.
The data itself must not change: only colours, highlights, extra summary rows, sorting,
hidden columns and the title. Use only the listed column names, spelled exactly.

- highlights: colour whole rows that match a rule. For "overdue"/"late" use the due date
  column with op "before_today"; "due this week" -> op "within_next_days" value 7.
  op "empty"/"not_empty" need no value. color is a colour name (red, green, yellow...).
- summary_rows: any of total, average, min, max, count (adds rows under the table).
- Start from the CURRENT settings and change only what the user asks; return them all.
Reply with the JSON only."""


@cache
def style_json_schema() -> dict[str, Any]:
    schema = ExportStyle.model_json_schema()
    defs = schema.pop("$defs", {})

    def inline(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                return inline(defs[node["$ref"].split("/")[-1]])
            return {k: inline(v) for k, v in node.items() if k != "title" or isinstance(v, dict)}
        if isinstance(node, list):
            return [inline(x) for x in node]
        return node

    result: dict[str, Any] = inline(schema)
    return result


def column_kinds(columns: list[str], rows: list[list[Any]]) -> dict[str, str]:
    kinds: dict[str, str] = {}
    for j, c in enumerate(columns):
        values = [r[j] for r in rows if r[j] is not None]
        if values and all(isinstance(v, int | float) and not isinstance(v, bool) for v in values):
            kinds[c] = "number"
        elif values and all(isinstance(v, str) and _as_date(v) for v in values):
            kinds[c] = "date"
        else:
            kinds[c] = "text"
    return kinds


def interpret_style(
    llm: LLMProvider,
    request: str,
    *,
    columns: list[str],
    rows: list[list[Any]],
    current: ExportStyle,
    today: date,
) -> tuple[ExportStyle, list[str], LLMCall]:
    """The new style for `request`, plus notes about anything that couldn't be applied."""
    kinds = column_kinds(columns, rows)
    described = "\n".join(
        f"- {c} ({kinds[c]}; e.g. {', '.join(str(r[j]) for r in rows[:3] if r[j] is not None)})"
        for j, c in enumerate(columns)
    )
    user = (
        f"Columns:\n{described}\n\nToday is {today.isoformat()}.\n"
        f"CURRENT settings: {current.model_dump_json(exclude_defaults=True)}\n\n"
        f"Request: {request}"
    )
    import time

    started = time.perf_counter()
    out = llm.complete(
        [ChatMessage("system", STYLE_SYSTEM), ChatMessage("user", user)],
        json_schema=style_json_schema(),
        max_tokens=500,
    )
    call = LLMCall(
        "plan", llm.name, llm.model, out.prompt_tokens, out.completion_tokens,
        out.latency_ms or int((time.perf_counter() - started) * 1000), True,
    )  # fmt: skip
    try:
        style = ExportStyle.model_validate(json.loads(out.text))
    except (ValueError, ValidationError) as exc:
        raise LLMUnavailableError("I couldn't understand that formatting request.") from exc
    style, notes = validate_style(style, columns)
    return style, notes, call


def _match_column(name: str, columns: list[str]) -> str | None:
    lowered = {c.lower(): c for c in columns}
    return name if name in columns else lowered.get(name.strip().lower())


def validate_style(style: ExportStyle, columns: list[str]) -> tuple[ExportStyle, list[str]]:
    """Drop anything that refers to missing columns or unknown colours, and say so."""
    notes: list[str] = []
    highlights = []
    for h in style.highlights:
        col = _match_column(h.column, columns)
        if col is None:
            notes.append(f"this table has no column “{h.column}” to highlight by")
            continue
        if color(h.color, strong=False) is None:
            notes.append(f"I don't know the colour “{h.color}”, so I used red")
            h = h.model_copy(update={"color": "red"})
        highlights.append(h.model_copy(update={"column": col}))
    sort = style.sort
    if sort is not None:
        col = _match_column(sort.column, columns)
        if col is None:
            notes.append(f"I couldn't sort by “{sort.column}”: it isn't in this table")
            sort = None
        else:
            sort = sort.model_copy(update={"column": col})
    hidden = [c for c in (_match_column(h, columns) for h in style.hide_columns) if c]
    header = style.header_color
    if header and color(header, strong=True) is None:
        notes.append(f"I don't know the colour “{header}” for the header")
        header = None
    return (
        style.model_copy(
            update={"highlights": highlights, "sort": sort, "hide_columns": hidden,
                    "header_color": header}
        ),
        notes,
    )  # fmt: skip


# --- applying it (code) ---------------------------------------------------------------


def _as_date(v: Any) -> date | None:
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    if isinstance(v, str) and re.match(r"^\d{4}-\d{2}-\d{2}", v):
        try:
            return date.fromisoformat(v[:10])
        except ValueError:
            return None
    return None


def _as_number(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, int | float):
        return float(v)
    try:
        return float(str(v).replace(",", ""))
    except ValueError:
        return None


def row_matches(h: Highlight, value: Any, today: date) -> bool:
    empty = value is None or (isinstance(value, str) and not value.strip())
    if h.op == "empty":
        return empty
    if h.op == "not_empty":
        return not empty
    if empty:
        return False
    if h.op in ("before_today", "after_today", "within_next_days"):
        d = _as_date(value)
        if d is None:
            return False
        if h.op == "before_today":
            return d < today
        if h.op == "after_today":
            return d > today
        days = _as_number(h.value) or 7
        return 0 <= (d - today).days <= days
    if h.op == "contains":
        return str(h.value or "").lower() in str(value).lower()
    left_n, right_n = _as_number(value), _as_number(h.value)
    left: Any
    right: Any
    if left_n is not None and right_n is not None:
        left, right = left_n, right_n
    elif (dl := _as_date(value)) and (dr := _as_date(h.value)):
        left, right = dl, dr
    else:
        left, right = str(value).strip().lower(), str(h.value or "").strip().lower()
    try:
        return bool(
            {"=": left == right, "!=": left != right, ">": left > right, ">=": left >= right,
             "<": left < right, "<=": left <= right}[h.op]
        )  # fmt: skip
    except TypeError:
        return False


@dataclass
class StyledTable:
    columns: list[str]
    rows: list[list[Any]]
    row_colors: list[str | None]  # highlight colour name per row
    summary: list[tuple[str, list[Any]]] = field(default_factory=list)  # (label, values)
    header: tuple[str, str] | None = None  # (fill, text)


def _summary_value(kind: str, values: list[float]) -> float | int | None:
    if not values:
        return None
    if kind == "total":
        return sum(values)
    if kind == "average":
        return sum(values) / len(values)
    if kind == "min":
        return min(values)
    if kind == "max":
        return max(values)
    return len(values)


def apply_style(
    columns: list[str], rows: list[list[Any]], style: ExportStyle, today: date,
    *, numeric_columns: list[str],
) -> StyledTable:  # fmt: skip
    """Sort, hide, highlight and summarise. Summary values are computed here, in code."""
    order = list(range(len(rows)))
    if style.sort and style.sort.column in columns:
        j = columns.index(style.sort.column)

        def key(i: int) -> tuple[int, Any]:
            v = rows[i][j]
            n, d = _as_number(v), _as_date(v)
            if v is None:
                return (1, 0)
            return (0, n if n is not None else (d.toordinal() if d else str(v).lower()))

        present = [i for i in order if rows[i][j] is not None]
        missing = [i for i in order if rows[i][j] is None]
        present.sort(key=key, reverse=style.sort.descending)
        order = present + missing  # blanks last either way
    keep = [j for j, c in enumerate(columns) if c not in style.hide_columns]
    out_cols = [columns[j] for j in keep]
    out_rows = [[rows[i][j] for j in keep] for i in order]

    row_colors: list[str | None] = []
    for i in order:
        hit = None
        for h in style.highlights:
            if h.column in columns and row_matches(h, rows[i][columns.index(h.column)], today):
                hit = h.color
                break  # first matching rule wins
        row_colors.append(hit)

    summary = []
    for kind in style.summary_rows or []:
        values = []
        for j, c in enumerate(out_cols):
            if c in numeric_columns:
                nums = [float(r[j]) for r in out_rows if isinstance(r[j], int | float)]
                values.append(_summary_value(kind, nums))
            else:
                values.append(None)
        summary.append((kind.capitalize(), values))
    return StyledTable(
        out_cols, out_rows, row_colors, summary, color(style.header_color, strong=True)
    )


def describe_style(style: ExportStyle) -> str:
    """What the export now looks like, in plain English (written by code)."""
    parts = []
    if style.header_color:
        parts.append(f"{style.header_color} header")
    for h in style.highlights:
        rule = {
            "before_today": "is before today", "after_today": "is after today",
            "within_next_days": f"is within the next {h.value or 7} days",
            "empty": "is empty", "not_empty": "is filled in",
        }.get(h.op, f"{h.op} {h.value}")  # fmt: skip
        parts.append(f"rows where {h.column} {rule} in {h.color}")
    if style.summary_rows:
        parts.append(
            " and ".join(style.summary_rows) + " row" + ("s" if len(style.summary_rows) > 1 else "")
        )
    if style.sort:
        direction = "highest/latest" if style.sort.descending else "lowest/earliest"
        parts.append(f"sorted by {style.sort.column} ({direction} first)")
    if style.hide_columns:
        parts.append("hidden: " + ", ".join(style.hide_columns))
    if style.title:
        parts.append(f"title “{style.title}”")
    return "; ".join(parts) if parts else "the standard AskData style"


def is_finite(v: Any) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)
