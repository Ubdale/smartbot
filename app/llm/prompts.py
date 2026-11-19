"""Prompts for planning and summarising. Kept short: small local models do better that way.

The few-shot examples use a made-up dataset on purpose, so the model learns the
*format* without memorising our sample data (which the golden eval uses).
"""

import json
from functools import cache
from typing import TYPE_CHECKING, Any

from app.schemas.query_plan import QueryPlan

if TYPE_CHECKING:
    from app.llm.base import ChatMessage

PLAN_SYSTEM = """You turn a business question into a JSON query plan for ONE table.
You never write SQL and never compute numbers; the backend does that.

Rules:
- Use only column names from the table description, spelled exactly.
- metric: count (rows), count_distinct, sum, avg, min, max. sum/avg need a number column.
  Use "rows" to list the matching rows themselves ("show/list ...", "which ... are",
  "all the formulas"), with columns = the columns worth showing (empty = all).
  Use "describe" for "what is this file/data about?" or "what columns are there?".
- group_by: at most 2 columns. Use time_grain (day/week/month/quarter/year) with
  date_column for trends over time ("by month", "monthly", "over time").
- date_phrase: copy the time period from the question word for word ("last quarter",
  "this year", "in 2025", "last 30 days"). Do NOT convert it to dates. Set date_column too.
- filters: op is one of = != > >= < <= in contains between. Use values from the
  column samples when possible. "in" takes a list; "between" takes [low, high].
- "top N" / "bottom N": sort by value desc/asc and set limit N.
- chart: bar for comparing groups, line for trends over time, pie for shares of a total
  (few groups), none for a single number.
- If the question is a follow-up ("now split by city", "only for 2025", "what about X"),
  copy the previous plan and change only what was asked: keep its metric, groups,
  filters and date_phrase. "Split by X" ADDS X to group_by.
- Prefer answering. Never ask back about what the file contains: use "describe". For vague
  requests ("show me the data", "all of it") use "rows". Only if the question is about
  something this table can't answer, set clarification_needed to a short question.
- If you asked a clarifying question and the user replies ("yes", "the formulas"), answer
  the original question using that reply. Never ask the same question twice.
Reply with the JSON plan only."""

EXAMPLE_TABLE = """Table "orders" (12,400 rows). Columns:
- order_day (date; 2023-01-02 to 2025-12-30)
- store (text; e.g. "Downtown", "Airport", "Harbor"; 6 distinct)
- product_type (text; e.g. "Coffee", "Tea", "Pastry"; 9 distinct)
- customer_id (text; e.g. "C0192", "C0441", "C0007"; 2,310 distinct)
- amount (number; 1.5 to 420)
- is_online (boolean)"""

EXAMPLES: list[tuple[str, dict[str, Any]]] = [
    ("How many orders were there last month?",
     {"metric": "count", "date_column": "order_day", "date_phrase": "last month",
      "chart": "none"}),
    ("Top 5 stores by revenue this year",
     {"metric": "sum", "metric_column": "amount", "group_by": ["store"],
      "date_column": "order_day", "date_phrase": "this year",
      "sort": {"by": "value", "direction": "desc"}, "limit": 5, "chart": "bar"}),
    ("Monthly sales trend for coffee in 2024",
     {"metric": "sum", "metric_column": "amount", "time_grain": "month",
      "date_column": "order_day", "date_phrase": "2024",
      "filters": [{"column": "product_type", "op": "=", "value": "Coffee"}],
      "limit": 100, "chart": "line"}),
    ("Average order value by store and channel",
     {"metric": "avg", "metric_column": "amount", "group_by": ["store", "is_online"],
      "limit": 50, "chart": "bar"}),
    ("How many unique customers bought tea or pastries in the last 90 days?",
     {"metric": "count_distinct", "metric_column": "customer_id",
      "date_column": "order_day", "date_phrase": "last 90 days",
      "filters": [{"column": "product_type", "op": "in", "value": ["Tea", "Pastry"]}],
      "chart": "none"}),
    ("Share of online vs in-store orders",
     {"metric": "count", "group_by": ["is_online"], "chart": "pie"}),
    ("Which orders are over 300?",
     {"metric": "count", "group_by": ["store"],
      "filters": [{"column": "amount", "op": ">", "value": 300}], "chart": "bar"}),
    ("List the online orders over 300 with their store",
     {"metric": "rows", "columns": ["order_day", "store", "amount"],
      "filters": [{"column": "is_online", "op": "=", "value": True},
                  {"column": "amount", "op": ">", "value": 300}],
      "sort": {"by": "amount", "direction": "desc"}, "limit": 50}),
    ("What is this data about?", {"metric": "describe"}),
    ("How is our employee morale?",
     {"clarification_needed": "This table has orders, not employee data. Do you want "
      "to ask about orders, stores or customers instead?"}),
]  # fmt: skip

# A follow-up turn: (previous question, previous plan, follow-up, new plan).
FOLLOW_UP_EXAMPLE: tuple[str, dict[str, Any], str, dict[str, Any]] = (
    "Revenue by store this year",
    {"metric": "sum", "metric_column": "amount", "group_by": ["store"],
     "date_column": "order_day", "date_phrase": "this year", "chart": "bar"},
    "now split by product type",
    {"metric": "sum", "metric_column": "amount", "group_by": ["store", "product_type"],
     "date_column": "order_day", "date_phrase": "this year", "limit": 50, "chart": "bar"},
)  # fmt: skip


@cache
def plan_json_schema() -> dict[str, Any]:
    """QueryPlan's JSON schema, inlined (no $refs) so local models' grammars handle it."""
    schema = QueryPlan.model_json_schema()
    defs = schema.pop("$defs", {})

    def inline(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                return inline(defs[node["$ref"].split("/")[-1]])
            return {k: inline(v) for k, v in node.items() if k != "title"}
        if isinstance(node, list):
            return [inline(x) for x in node]
        return node

    result: dict[str, Any] = inline(schema)
    return result


def compact_plan(plan: dict[str, Any] | QueryPlan) -> str:
    """A plan as short JSON (defaults omitted): fewer tokens, which matters on CPU."""
    model = plan if isinstance(plan, QueryPlan) else QueryPlan.model_validate(plan)
    return json.dumps(model.model_dump(mode="json", exclude_defaults=True), separators=(",", ":"))


@cache
def plan_system_prompt() -> str:
    """Rules plus few-shot examples, all in one message. The example table is shown once."""
    examples = "\n".join(f"Q: {q}\nA: {compact_plan(p)}" for q, p in EXAMPLES)
    prev_q, prev_plan, follow_q, follow_plan = FOLLOW_UP_EXAMPLE
    follow_up = (
        f"Q: {prev_q}\nA: {compact_plan(prev_plan)}\n"
        f"Q (follow-up): {follow_q}\nA: {compact_plan(follow_plan)}"
    )
    return (
        f"{PLAN_SYSTEM}\n\nExamples for a different table (fields left out use defaults):\n"
        f"{EXAMPLE_TABLE}\n\n{examples}\n\nFollow-up example:\n{follow_up}"
    )


def plan_messages(
    question: str, schema_text: str, history: list["ChatMessage"]
) -> list["ChatMessage"]:
    from app.llm.base import ChatMessage

    return [
        ChatMessage("system", plan_system_prompt()),
        *history,
        ChatMessage("user", f"{schema_text}\n\nQuestion: {question}"),
    ]


def retry_instruction(problem: str) -> str:
    return (
        f"That plan can't be used: {problem}\n"
        "Reply with a corrected JSON plan for the same question, using only the listed columns."
    )


SUMMARY_SYSTEM = """You write a 2-3 sentence plain-English summary of a query result for a
business user. Use ONLY numbers that appear in the result table, copied exactly or rounded
(e.g. 1,234,567 -> 1.2 million). Never calculate new numbers: no sums, differences,
percentages or averages of your own. Mention the most important finding first. No
preamble, no bullet points, no markdown."""


def summary_messages(question: str, description: str, table_text: str) -> list["ChatMessage"]:
    from app.llm.base import ChatMessage

    return [
        ChatMessage("system", SUMMARY_SYSTEM),
        ChatMessage(
            "user",
            f"Question: {question}\nWhat was computed: {description}\n\nResult:\n{table_text}",
        ),
    ]


TABLE_FROM_IMAGE = """Copy the table in this image into rows of cells, exactly as shown.
- The first row is the header row (column names). If the table has title lines above the
  header, leave them out.
- Copy every number exactly as printed (keep commas, decimals, signs and units). Never
  calculate, round, guess or fill in values; an empty or unreadable cell is "".
- One inner list per table row, cells left to right, same number of cells in every row.
- If there are several tables, copy the largest one. If there is no table, return no rows."""

TABLE_ROWS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "rows": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}}
    },
    "required": ["rows"],
}


DOC_SYSTEM = """You answer questions about a document using ONLY the numbered passages given.
- Answer in 1-4 sentences of plain English. Cite the passages you used like [1] or [2][3].
- Copy numbers, dates and names exactly as written in the passages. Never calculate,
  estimate or add facts that aren't in the passages.
- If the passages don't contain the answer, set found to false and say briefly that the
  document doesn't seem to cover it."""

DOC_ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "found": {"type": "boolean"},
        "answer": {"type": "string"},
        "sources": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["found", "answer", "sources"],
}


def doc_messages(
    question: str, passages: list[tuple[int, int | None, str]], history: list["ChatMessage"]
) -> list["ChatMessage"]:
    """passages: (number shown to the model, page, text)."""
    from app.llm.base import ChatMessage

    blocks = "\n\n".join(
        f"[{n}]{f' (page {page})' if page else ''}\n{text}" for n, page, text in passages
    )
    return [
        ChatMessage("system", DOC_SYSTEM),
        *history,
        ChatMessage("user", f"Passages:\n{blocks}\n\nQuestion: {question}"),
    ]
