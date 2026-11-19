"""Validate a QueryPlan against a dataset and run it with DuckDB.

Safety rules:
- Every column must exist in the dataset's profile (exact, case-insensitive match).
- SQL is assembled only from fixed templates; identifiers are quoted and every
  value (from the user or the LLM) is passed as a DuckDB parameter.
- Queries run with a timeout and a row cap, and results are cached by
  (dataset id, dataset version, plan).
"""

import hashlib
import json
import threading
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import duckdb
from cachetools import TTLCache
from rapidfuzz import process

from app.connectors.profiling import jsonable
from app.query.dates import DatePhraseError, DateRange, resolve_date_phrase
from app.query.sql import quote_ident
from app.schemas.query_plan import Filter, QueryPlan, Scalar

QUERY_TIMEOUT_SECONDS = 10.0
MAX_IN_VALUES = 100


class PlanError(ValueError):
    """The plan doesn't fit the dataset. The message is written for the LLM and the user."""


class QueryTimeoutError(RuntimeError):
    pass


@dataclass(frozen=True)
class ColumnInfo:
    name: str
    kind: str  # number | text | date | datetime | boolean


@dataclass(frozen=True)
class ValidatedPlan:
    """A plan whose columns are canonical and whose dates are resolved."""

    plan: QueryPlan
    kinds: dict[str, str]
    date_range: DateRange | None
    filter_values: list[Any]  # coerced value per filter, same order as plan.filters

    @property
    def dimension_aliases(self) -> list[str]:
        if self.plan.metric in ("rows", "describe"):
            return []
        dims: list[str] = [self.plan.time_grain] if self.plan.time_grain else []
        return dims + list(self.plan.group_by)

    @property
    def metric_alias(self) -> str:
        p = self.plan
        if p.metric in ("rows", "describe"):
            return ""
        alias = "count" if p.metric == "count" else f"{p.metric}_{p.metric_column}"
        lowered = {d.lower() for d in self.dimension_aliases}
        while alias.lower() in lowered:  # e.g. grouping by a column literally named "count"
            alias = f"{alias}_value"
        return alias


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[list[Any]]
    truncated: bool  # more groups exist than `limit`
    sql: str
    params: list[Any]
    dimensions: list[str]
    value_column: str
    cached: bool = False
    elapsed_ms: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


# --- validation -----------------------------------------------------------------------


def _suggest(name: str, columns: list[ColumnInfo]) -> str:
    matches = process.extract(name, [c.name for c in columns], limit=3, score_cutoff=55)
    if matches:
        return " Did you mean: " + ", ".join(repr(m[0]) for m in matches) + "?"
    return " Available columns: " + ", ".join(repr(c.name) for c in columns[:40]) + "."


def _resolve_column(name: str, columns: list[ColumnInfo], role: str) -> ColumnInfo:
    for c in columns:
        if c.name == name:
            return c
    lowered = [c for c in columns if c.name.lower() == name.strip().lower()]
    if len(lowered) == 1:
        return lowered[0]
    raise PlanError(f"Unknown column {name!r} in {role}.{_suggest(name, columns)}")


def _coerce(value: Scalar, kind: str, column: str) -> Any:
    try:
        if kind == "number":
            if isinstance(value, bool):
                raise ValueError
            if isinstance(value, int | float):
                return value
            text = str(value).replace(",", "").strip()
            return int(text) if text.lstrip("-").isdigit() else float(text)
        if kind == "date":
            return date.fromisoformat(str(value)[:10])
        if kind == "datetime":
            return datetime.fromisoformat(str(value))
        if kind == "boolean":
            if isinstance(value, bool):
                return value
            text = str(value).strip().lower()
            if text in ("true", "yes", "1", "y"):
                return True
            if text in ("false", "no", "0", "n"):
                return False
            raise ValueError
        return str(value)
    except ValueError:
        raise PlanError(f"Value {value!r} doesn't fit column {column!r} ({kind}).") from None


def _validate_filter(f: Filter, col: ColumnInfo) -> Any:
    kind, op, v = col.kind, f.op, f.value
    if v is None:
        if op not in ("=", "!="):
            raise PlanError(f"Filter on {col.name!r}: operator {op!r} needs a value.")
        return None
    if op == "in":
        values = v if isinstance(v, list) else [v]
        if not values or len(values) > MAX_IN_VALUES:
            raise PlanError(f"Filter 'in' on {col.name!r} needs 1-{MAX_IN_VALUES} values.")
        return [_coerce(x, kind, col.name) for x in values]
    if op == "between":
        if not isinstance(v, list) or len(v) != 2:
            raise PlanError(f"Filter 'between' on {col.name!r} needs [low, high].")
        if kind not in ("number", "date", "datetime"):
            raise PlanError(f"'between' needs a number or date column, not {col.name!r}.")
        return [_coerce(v[0], kind, col.name), _coerce(v[1], kind, col.name)]
    if isinstance(v, list):
        raise PlanError(f"Filter {op!r} on {col.name!r} takes a single value, not a list.")
    if op == "contains":
        if kind != "text":
            raise PlanError(f"'contains' only works on text columns, not {col.name!r}.")
        return str(v)
    if op in (">", ">=", "<", "<=") and kind not in ("number", "date", "datetime"):
        raise PlanError(f"{op!r} needs a number or date column, not {col.name!r} ({kind}).")
    return _coerce(v, kind, col.name)


def validate_plan(plan: QueryPlan, columns: list[ColumnInfo], today: date) -> ValidatedPlan:
    """Check every column and operator against the dataset. Raises PlanError."""
    if plan.clarification_needed:
        raise PlanError("A plan that needs clarification cannot be executed.")
    kinds: dict[str, str] = {}

    def col(name: str, role: str) -> ColumnInfo:
        c = _resolve_column(name, columns, role)
        kinds[c.name] = c.kind
        return c

    metric_column = None
    list_columns: list[str] = []
    if plan.metric == "describe":
        return ValidatedPlan(QueryPlan(metric="describe"), {}, None, [])
    if plan.metric == "rows":
        if plan.group_by or plan.time_grain:
            raise PlanError("metric 'rows' lists rows: leave group_by and time_grain empty.")
        list_columns = [col(c, "columns").name for c in plan.columns]
    elif plan.metric == "count":
        pass  # count(*) ignores metric_column
    else:
        if not plan.metric_column:
            raise PlanError(f"Metric {plan.metric!r} needs a metric_column.")
        mc = col(plan.metric_column, "metric_column")
        if plan.metric in ("sum", "avg") and mc.kind != "number":
            raise PlanError(
                f"Can't {plan.metric} {mc.name!r}: it is {mc.kind}, not a number. "
                "Use count or count_distinct instead."
            )
        if plan.metric in ("min", "max") and mc.kind not in ("number", "date", "datetime"):
            raise PlanError(f"{plan.metric} needs a number or date column, not {mc.name!r}.")
        metric_column = mc.name

    group_by = [col(g, "group_by").name for g in plan.group_by]
    if len(set(group_by)) != len(group_by):
        raise PlanError("group_by lists the same column twice.")

    date_column = None
    if plan.date_column:
        dc = col(plan.date_column, "date_column")
        if dc.kind not in ("date", "datetime"):
            raise PlanError(f"date_column {dc.name!r} is {dc.kind}, not a date.")
        date_column = dc.name
    elif plan.date_phrase or plan.time_grain:
        date_cols = [c for c in columns if c.kind in ("date", "datetime")]
        if len(date_cols) != 1:
            names = ", ".join(repr(c.name) for c in date_cols) or "none"
            raise PlanError(f"Set date_column to one of the date columns: {names}.")
        date_column = date_cols[0].name
        kinds[date_column] = date_cols[0].kind

    if plan.time_grain:
        if len(group_by) > 1:
            raise PlanError("With time_grain, group_by can have at most one column.")
        if plan.time_grain in group_by:
            raise PlanError(f"{plan.time_grain!r} is reserved for the time grouping.")

    date_range = None
    if plan.date_phrase:
        try:
            date_range = resolve_date_phrase(plan.date_phrase, today)
        except DatePhraseError:
            raise PlanError(
                f"Couldn't understand the time period {plan.date_phrase!r}. Use a phrase like "
                "'last quarter', 'this year', 'last 30 days', 'Q3 2025', 'March 2025' or "
                "'2025-01-01 to 2025-03-31'."
            ) from None

    filters, filter_values = [], []
    for f in plan.filters:
        c = col(f.column, "filters")
        filter_values.append(_validate_filter(f, c))
        filters.append(f.model_copy(update={"column": c.name}))

    dims = ([plan.time_grain] if plan.time_grain else []) + group_by
    if plan.metric == "rows":
        dims = [c.name for c in columns]  # any column can order a list of rows
    sort = plan.sort
    # Models often name the measured column ("revenue", "sum_revenue") meaning "the value".
    value_names = {"value", plan.metric, f"{plan.metric}_{metric_column}", metric_column or ""}
    if sort and sort.by not in dims and sort.by.lower() in {v.lower() for v in value_names if v}:
        sort = sort.model_copy(update={"by": "value"})
    if sort and sort.by != "value" and sort.by not in dims:
        match = next((d for d in dims if d.lower() == sort.by.lower()), None)
        if match is None:
            raise PlanError(
                f"sort.by must be 'value' or one of the groups {dims}, not {sort.by!r}."
            )
        sort = sort.model_copy(update={"by": match})

    canonical = plan.model_copy(
        update={
            "metric_column": metric_column,
            "columns": list_columns,
            "group_by": group_by,
            "date_column": date_column,
            "filters": filters,
            "sort": sort,
        }
    )
    return ValidatedPlan(canonical, kinds, date_range, filter_values)


# --- SQL ------------------------------------------------------------------------------

_COMPARE = {"=": "=", "!=": "<>", ">": ">", ">=": ">=", "<": "<", "<=": "<="}
_METRIC_SQL = {"sum": "sum", "avg": "avg", "min": "min", "max": "max"}
_GRAINS = {"day", "week", "month", "quarter", "year"}


def _escape_like(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _filter_sql(f: Filter, kind: str, value: Any, params: list[Any]) -> str:
    q = quote_ident(f.column)
    text = kind == "text"
    if value is None:
        return f"{q} IS NULL" if f.op == "=" else f"{q} IS NOT NULL"
    if f.op == "contains":
        params.append(f"%{_escape_like(value)}%")
        return f"CAST({q} AS VARCHAR) ILIKE ? ESCAPE '\\'"
    if f.op == "between":
        params.extend(value)
        return f"{q} BETWEEN ? AND ?"
    if f.op == "in":
        params.extend(value)
        marks = ", ".join("lower(?)" if text else "?" for _ in value)
        return f"lower({q}) IN ({marks})" if text else f"{q} IN ({marks})"
    params.append(value)
    op = _COMPARE[f.op]
    if text and f.op in ("=", "!="):
        return f"lower({q}) {op} lower(?)"  # 'north' matches 'North'
    return f"{q} {op} ?"


def _where(vp: ValidatedPlan, params: list[Any]) -> list[str]:
    p = vp.plan
    where: list[str] = []
    if vp.date_range and p.date_column:
        dq = quote_ident(p.date_column)
        where.append(f"{dq} >= ? AND {dq} < ?")
        params += [vp.date_range.start, vp.date_range.end]
    for f, value in zip(p.filters, vp.filter_values, strict=True):
        where.append(_filter_sql(f, vp.kinds[f.column], value, params))
    return where


def _rows_sql(vp: ValidatedPlan, parquet: Path) -> tuple[str, list[Any]]:
    """metric "rows": the matching rows themselves (selected columns, filtered, capped)."""
    p = vp.plan
    params: list[Any] = [str(parquet)]
    select = ", ".join(quote_ident(c) for c in p.columns) if p.columns else "*"
    sql = f"SELECT {select}\nFROM read_parquet(?)"
    where = _where(vp, params)
    if where:
        sql += "\nWHERE " + "\n  AND ".join(where)
    if p.sort and p.sort.by != "value":
        direction = "ASC" if p.sort.direction == "asc" else "DESC"
        sql += f"\nORDER BY {quote_ident(p.sort.by)} {direction} NULLS LAST"
    sql += "\nLIMIT ?"
    params.append(p.limit + 1)
    return sql, params


def build_sql(vp: ValidatedPlan, parquet: Path) -> tuple[str, list[Any]]:
    p = vp.plan
    if p.metric == "rows":
        return _rows_sql(vp, parquet)
    params: list[Any] = [str(parquet)]

    select: list[str] = []
    if p.time_grain:
        if p.time_grain not in _GRAINS or not p.date_column:
            raise PlanError("time_grain needs a date column.")
        expr = f"CAST(date_trunc('{p.time_grain}', {quote_ident(p.date_column)}) AS DATE)"
        select.append(f"{expr} AS {quote_ident(p.time_grain)}")
    select += [quote_ident(g) for g in p.group_by]

    if p.metric == "count":
        metric = "count(*)"
    elif p.metric == "count_distinct":
        metric = f"count(DISTINCT {quote_ident(p.metric_column or '')})"
    else:
        metric = f"{_METRIC_SQL[p.metric]}({quote_ident(p.metric_column or '')})"
    select.append(f"{metric} AS {quote_ident(vp.metric_alias)}")

    where = _where(vp, params)

    sql = f"SELECT {', '.join(select)}\nFROM read_parquet(?)"
    if where:
        sql += "\nWHERE " + "\n  AND ".join(where)

    dims = vp.dimension_aliases
    if dims:
        sql += "\nGROUP BY ALL"
        if p.sort:
            primary = quote_ident(vp.metric_alias if p.sort.by == "value" else p.sort.by)
            direction = "ASC" if p.sort.direction == "asc" else "DESC"
        elif p.time_grain:
            primary, direction = quote_ident(p.time_grain), "ASC"
        else:
            primary, direction = quote_ident(vp.metric_alias), "DESC"
        tiebreak = ", ".join(f"{quote_ident(d)} ASC NULLS LAST" for d in dims)
        sql += f"\nORDER BY {primary} {direction} NULLS LAST, {tiebreak}"
        sql += "\nLIMIT ?"
        params.append(p.limit + 1)  # one extra row tells us the result was truncated
    return sql, params


# --- execution + cache ----------------------------------------------------------------

_cache: TTLCache[str, QueryResult] = TTLCache(maxsize=512, ttl=600)
_cache_lock = threading.Lock()


def _cache_key(dataset_id: str, version: int, vp: ValidatedPlan) -> str:
    payload = {
        "plan": vp.plan.model_dump(mode="json"),
        "range": [str(vp.date_range.start), str(vp.date_range.end)] if vp.date_range else None,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return f"{dataset_id}:{version}:{digest}"


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()


def execute_plan(
    vp: ValidatedPlan,
    parquet: Path,
    *,
    dataset_id: str,
    version: int,
    timeout: float = QUERY_TIMEOUT_SECONDS,
) -> QueryResult:
    key = _cache_key(dataset_id, version, vp)
    with _cache_lock:
        hit = _cache.get(key)
    if hit is not None:
        return QueryResult(**{**hit.__dict__, "cached": True})

    sql, params = build_sql(vp, parquet)
    started = datetime.now()
    with duckdb.connect() as con:
        con.execute("SET threads TO 2")
        timer = threading.Timer(timeout, con.interrupt)
        timer.start()
        try:
            cur = con.execute(sql, params)
            columns = [d[0] for d in cur.description or []]
            raw = cur.fetchmany(vp.plan.limit + 1)
        except duckdb.InterruptException as exc:
            raise QueryTimeoutError(f"The query took longer than {timeout:g}s.") from exc
        finally:
            timer.cancel()
    elapsed = int((datetime.now() - started).total_seconds() * 1000)

    truncated = len(raw) > vp.plan.limit
    rows = [[jsonable(v) for v in r] for r in raw[: vp.plan.limit]]
    result = QueryResult(
        columns=columns,
        rows=rows,
        truncated=truncated,
        sql=sql,
        params=[jsonable(p) for p in params[1:]],  # hide the server file path
        dimensions=vp.dimension_aliases,
        value_column=vp.metric_alias,
        elapsed_ms=elapsed,
    )
    with _cache_lock:
        _cache[key] = result
    return result


def columns_from_profile(profile: list[dict[str, Any]]) -> list[ColumnInfo]:
    return [ColumnInfo(name=str(c["name"]), kind=str(c["kind"])) for c in profile]
