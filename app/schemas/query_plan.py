"""The query plan: the *only* thing the LLM produces when interpreting a question.

The backend validates it against the dataset's columns and turns it into
parameterised DuckDB SQL. The LLM never writes SQL and never invents numbers.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# "rows" lists matching rows themselves; "describe" answers "what is this data about?".
Metric = Literal["count", "count_distinct", "sum", "avg", "min", "max", "rows", "describe"]
FilterOp = Literal["=", "!=", ">", ">=", "<", "<=", "in", "contains", "between"]
TimeGrain = Literal["day", "week", "month", "quarter", "year"]
ChartType = Literal["none", "bar", "line", "pie"]
Scalar = str | int | float | bool

MAX_GROUP_BY = 2
MAX_FILTERS = 10
MAX_LIMIT = 100
MAX_LIST_COLUMNS = 20


class Filter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: str
    op: FilterOp
    # A single value; a list for "in"; [low, high] for "between"; null means "is empty".
    value: Scalar | list[Scalar] | None = None


class Sort(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # "value" (the metric) or the name of a group_by column (or the time_grain name).
    by: str = "value"
    direction: Literal["asc", "desc"] = "desc"


class QueryPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Fill this (and nothing else matters) when the question is too ambiguous to answer.
    clarification_needed: str | None = None
    metric: Metric = "count"
    metric_column: str | None = None
    # For metric "rows": the columns to show (empty = all).
    columns: list[str] = Field(default_factory=list, max_length=MAX_LIST_COLUMNS)
    group_by: list[str] = Field(default_factory=list, max_length=MAX_GROUP_BY)
    # Group the date_column into periods, e.g. "revenue by month".
    time_grain: TimeGrain | None = None
    date_column: str | None = None
    # The time period exactly as phrased, e.g. "last quarter"; code turns it into dates.
    date_phrase: str | None = None
    filters: list[Filter] = Field(default_factory=list, max_length=MAX_FILTERS)
    sort: Sort | None = None
    limit: int = Field(default=20, ge=1, le=MAX_LIMIT)
    chart: ChartType = "none"
