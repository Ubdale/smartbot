"""Turn an answered plan into runnable pandas code (written by code, not by the model).

Lets users reproduce or extend an answer themselves: same filters, same date range,
same grouping and sorting as what AskData computed.
"""

from typing import Any

from app.schemas.query_plan import QueryPlan

_READERS = {
    "csv": "pd.read_csv({path!r})",
    "tsv": "pd.read_csv({path!r}, sep='\\t')",
    "txt": "pd.read_csv({path!r}, sep=None, engine='python')",
    "xlsx": "pd.read_excel({path!r})",
    "xls": "pd.read_excel({path!r})",
    "ods": "pd.read_excel({path!r})",
    "json": "pd.read_json({path!r})",
    "parquet": "pd.read_parquet({path!r})",
}
_PERIODS = {"day": "D", "week": "W", "month": "M", "quarter": "Q", "year": "Y"}


def _col(name: str) -> str:
    return f"df[{name!r}]"


def _filter_line(f: dict[str, Any]) -> str:
    c, op, v = _col(f["column"]), f["op"], f.get("value")
    if v is None:
        return f"df = df[{c}.isna()]" if op == "=" else f"df = df[{c}.notna()]"
    if op == "contains":
        return f"df = df[{c}.astype(str).str.contains({str(v)!r}, case=False, regex=False)]"
    if op == "in":
        values = v if isinstance(v, list) else [v]
        if all(isinstance(x, str) for x in values):
            lowered = [x.lower() for x in values]
            return f"df = df[{c}.astype(str).str.lower().isin({lowered!r})]"
        return f"df = df[{c}.isin({values!r})]"
    if op == "between":
        return f"df = df[{c}.between({v[0]!r}, {v[1]!r})]"
    if isinstance(v, str) and op in ("=", "!="):
        cmp = "==" if op == "=" else "!="
        return f"df = df[{c}.astype(str).str.lower() {cmp} {v.lower()!r}]"
    return f"df = df[{c} {op if op != '=' else '=='} {v!r}]"


def pandas_code(
    plan_json: dict[str, Any],
    *,
    file_name: str,
    file_format: str,
    date_range: dict[str, str] | None,
) -> str:
    plan = QueryPlan.model_validate(plan_json)
    reader = _READERS.get(file_format, "pd.read_csv({path!r})").format(path=file_name)
    lines = ["import pandas as pd", "", f"df = {reader}"]

    if plan.metric == "describe":
        lines += ["", "print(df.dtypes)", "print(df.describe(include='all'))"]
        return "\n".join(lines) + "\n"

    if plan.date_column and (date_range or plan.time_grain):
        lines.append(f"{_col(plan.date_column)} = pd.to_datetime({_col(plan.date_column)})")
    if date_range and plan.date_column:
        lines.append(
            f"# {date_range['label']}: {date_range['start']} to {date_range['end']} (inclusive)"
        )
        lines.append(
            f"df = df[({_col(plan.date_column)} >= {date_range['start']!r}) & "
            f"({_col(plan.date_column)} <= {date_range['end'] + ' 23:59:59'!r})]"
        )
    lines += [_filter_line(f.model_dump()) for f in plan.filters]

    if plan.metric == "rows":
        cols = f"[{', '.join(repr(c) for c in plan.columns)}]" if plan.columns else None
        result = f"df[{cols}]" if cols else "df"
        if plan.sort and plan.sort.by != "value":
            asc = plan.sort.direction == "asc"
            result += f".sort_values({plan.sort.by!r}, ascending={asc})"
        lines += ["", f"result = {result}.head({plan.limit})", "print(result)"]
        return "\n".join(lines) + "\n"

    keys = list(plan.group_by)
    if plan.time_grain and plan.date_column:
        grain = plan.time_grain
        lines.append(
            f"df[{grain!r}] = {_col(plan.date_column)}.dt.to_period("
            f"{_PERIODS[grain]!r}).dt.start_time.dt.date"
        )
        keys = [grain, *keys]
    alias = "count" if plan.metric == "count" else f"{plan.metric}_{plan.metric_column}"
    agg = {
        "count": "size()",
        "count_distinct": f"[{plan.metric_column!r}].nunique()",
        "sum": f"[{plan.metric_column!r}].sum()",
        "avg": f"[{plan.metric_column!r}].mean()",
        "min": f"[{plan.metric_column!r}].min()",
        "max": f"[{plan.metric_column!r}].max()",
    }[plan.metric]

    lines.append("")
    if not keys:
        expr = "len(df)" if plan.metric == "count" else f"df{agg}"
        lines += [f"{alias} = {expr}", f"print({alias})"]
        return "\n".join(lines) + "\n"

    lines.append(f"result = df.groupby({keys!r}, dropna=False){agg}.reset_index(name={alias!r})")
    if plan.sort:
        by = alias if plan.sort.by == "value" else plan.sort.by
        lines.append(
            f"result = result.sort_values({by!r}, ascending={plan.sort.direction == 'asc'})"
        )
    elif plan.time_grain:
        lines.append(f"result = result.sort_values({plan.time_grain!r})")
    else:
        lines.append(f"result = result.sort_values({alias!r}, ascending=False)")
    lines += [f"result = result.head({plan.limit})", "print(result)"]
    return "\n".join(lines) + "\n"
