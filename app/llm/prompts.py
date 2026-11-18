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
  change the previous plan instead of starting over.
- If the question can't be answered from this table, or is truly ambiguous, set
  clarification_needed to a short question for the user and leave the rest default.
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
    ("How is our employee morale?",
     {"clarification_needed": "This table has orders, not employee data. Do you want "
      "to ask about orders, stores or customers instead?"}),
]  # fmt: skip


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


def _example_messages() -> list["ChatMessage"]:
    from app.llm.base import ChatMessage

    msgs = []
    for question, plan in EXAMPLES:
        full = QueryPlan.model_validate(plan).model_dump(mode="json")
        msgs.append(ChatMessage("user", f"{EXAMPLE_TABLE}\n\nQuestion: {question}"))
        msgs.append(ChatMessage("assistant", json.dumps(full, separators=(",", ":"))))
    return msgs


def plan_messages(
    question: str, schema_text: str, history: list["ChatMessage"]
) -> list["ChatMessage"]:
    from app.llm.base import ChatMessage

    return [
        ChatMessage("system", PLAN_SYSTEM),
        *_example_messages(),
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
