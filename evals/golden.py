"""Golden questions over the sample data, with answers computed independently in pandas.

Each case has: the question, the dataset, the expected answer (pandas), how to compare,
and a reference plan. The reference plans are checked by the test suite (proving the
executor computes correct numbers); `scripts/run_eval.py` checks what a real model plans.
Fixed "today" so relative dates ("last quarter") are reproducible.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

import pandas as pd

SAMPLE = Path(__file__).resolve().parent.parent / "sample_data"
TODAY = date(2026, 11, 19)
Compare = Literal["scalar", "groups", "ordered_keys", "first_key"]


@dataclass(frozen=True)
class Case:
    dataset: Literal["sales", "hr"]
    question: str
    expected: Callable[[pd.DataFrame], Any]
    compare: Compare
    plan: dict[str, Any]


def _sales() -> pd.DataFrame:
    return pd.read_csv(SAMPLE / "sales_orders.csv", parse_dates=["order_date"])


def _hr() -> pd.DataFrame:
    df = pd.read_csv(SAMPLE / "hr_employees.csv", parse_dates=["hire_date", "termination_date"])
    return df


def frame(dataset: str) -> pd.DataFrame:
    return _sales() if dataset == "sales" else _hr()


def _between(df: pd.DataFrame, col: str, start: str, end: str) -> pd.DataFrame:
    return df[(df[col] >= start) & (df[col] < end)]


def _month_key(s: pd.Series) -> pd.Series:
    keys: pd.Series = s.dt.to_period("M").dt.start_time.dt.strftime("%Y-%m-%d")
    return keys


SUM_REV = {"metric": "sum", "metric_column": "revenue"}

CASES: list[Case] = [
    # --- sales -------------------------------------------------------------------------
    Case("sales", "What is the total revenue?", lambda d: d.revenue.sum(), "scalar", SUM_REV),
    Case("sales", "How many orders are there?", lambda d: len(d), "scalar", {"metric": "count"}),
    Case("sales", "Revenue by region", lambda d: d.groupby("region").revenue.sum(), "groups",
         {**SUM_REV, "group_by": ["region"]}),
    Case("sales", "Top 5 customers by revenue",
         lambda d: d.groupby("customer").revenue.sum().nlargest(5), "ordered_keys",
         {**SUM_REV, "group_by": ["customer"], "sort": {"by": "value", "direction": "desc"},
          "limit": 5}),
    Case("sales", "Number of orders by category", lambda d: d.groupby("category").size(),
         "groups", {"metric": "count", "group_by": ["category"]}),
    Case("sales", "Average discount by product", lambda d: d.groupby("product").discount.mean(),
         "groups", {"metric": "avg", "metric_column": "discount", "group_by": ["product"]}),
    Case("sales", "What was the revenue last quarter?",
         lambda d: _between(d, "order_date", "2026-07-01", "2026-10-01").revenue.sum(), "scalar",
         {**SUM_REV, "date_column": "order_date", "date_phrase": "last quarter"}),
    Case("sales", "How many orders were placed in 2025?",
         lambda d: len(_between(d, "order_date", "2025-01-01", "2026-01-01")), "scalar",
         {"metric": "count", "date_column": "order_date", "date_phrase": "2025"}),
    Case("sales", "Monthly revenue in 2025",
         lambda d: (lambda x: x.groupby(_month_key(x.order_date)).revenue.sum())(
             _between(d, "order_date", "2025-01-01", "2026-01-01")), "groups",
         {**SUM_REV, "time_grain": "month", "date_column": "order_date", "date_phrase": "2025",
          "limit": 100}),
    Case("sales", "Revenue by region this year",
         lambda d: _between(d, "order_date", "2026-01-01", "2027-01-01").groupby("region")
         .revenue.sum(), "groups",
         {**SUM_REV, "group_by": ["region"], "date_column": "order_date",
          "date_phrase": "this year"}),
    Case("sales", "Which product sold the most units?",
         lambda d: d.groupby("product").quantity.sum().nlargest(1), "first_key",
         {"metric": "sum", "metric_column": "quantity", "group_by": ["product"],
          "sort": {"by": "value", "direction": "desc"}, "limit": 1}),
    Case("sales", "What is the average order value?", lambda d: d.revenue.mean(), "scalar",
         {"metric": "avg", "metric_column": "revenue"}),
    Case("sales", "How many different customers do we have?", lambda d: d.customer.nunique(),
         "scalar", {"metric": "count_distinct", "metric_column": "customer"}),
    Case("sales", "Subscriptions revenue by region",
         lambda d: d[d.category == "Subscriptions"].groupby("region").revenue.sum(), "groups",
         {**SUM_REV, "group_by": ["region"],
          "filters": [{"column": "category", "op": "=", "value": "Subscriptions"}]}),
    Case("sales", "How many orders had revenue over 10,000?",
         lambda d: int((d.revenue > 10000).sum()), "scalar", {"metric": "count",
                    "filters": [{"column": "revenue", "op": ">", "value": 10000}]}),
    Case("sales", "Total quantity sold in Karachi", lambda d: d[d.city == "Karachi"].quantity.sum(),
         "scalar", {"metric": "sum", "metric_column": "quantity",
                    "filters": [{"column": "city", "op": "=", "value": "Karachi"}]}),
    Case("sales", "Which city has the lowest revenue?",
         lambda d: d.groupby("city").revenue.sum().nsmallest(1), "first_key",
         {**SUM_REV, "group_by": ["city"], "sort": {"by": "value", "direction": "asc"},
          "limit": 1}),
    Case("sales", "Revenue by category and region",
         lambda d: d.groupby(["category", "region"]).revenue.sum(), "groups",
         {**SUM_REV, "group_by": ["category", "region"], "limit": 100}),
    Case("sales", "What was the biggest single order revenue?", lambda d: d.revenue.max(),
         "scalar", {"metric": "max", "metric_column": "revenue"}),
    Case("sales", "How many Training Day orders were there in 2026?",
         lambda d: len(_between(d[d["product"] == "Training Day"], "order_date", "2026-01-01",
                                "2027-01-01")), "scalar",
         {"metric": "count", "date_column": "order_date", "date_phrase": "2026",
          "filters": [{"column": "product", "op": "=", "value": "Training Day"}]}),
    # --- hr ----------------------------------------------------------------------------
    Case("hr", "Number of employees by department", lambda d: d.groupby("department").size(),
         "groups", {"metric": "count", "group_by": ["department"]}),
    Case("hr", "Average salary by job level", lambda d: d.groupby("job_level").annual_salary.mean(),
         "groups", {"metric": "avg", "metric_column": "annual_salary", "group_by": ["job_level"]}),
    Case("hr", "How many employees work remotely?", lambda d: int(d.is_remote.sum()), "scalar",
         {"metric": "count", "filters": [{"column": "is_remote", "op": "=", "value": True}]}),
    Case("hr", "How many people were hired in 2025?",
         lambda d: len(_between(d, "hire_date", "2025-01-01", "2026-01-01")), "scalar",
         {"metric": "count", "date_column": "hire_date", "date_phrase": "2025"}),
    Case("hr", "What is the average salary in Engineering?",
         lambda d: d[d.department == "Engineering"].annual_salary.mean(), "scalar",
         {"metric": "avg", "metric_column": "annual_salary",
          "filters": [{"column": "department", "op": "=", "value": "Engineering"}]}),
    Case("hr", "Employees per city", lambda d: d.groupby("city").size(), "groups",
         {"metric": "count", "group_by": ["city"]}),
    Case("hr", "How many employees have left the company?",
         lambda d: int(d.termination_date.notna().sum()), "scalar",
         {"metric": "count", "filters": [{"column": "termination_date", "op": "!="}]}),
    Case("hr", "Highest salary in each department",
         lambda d: d.groupby("department").annual_salary.max(), "groups",
         {"metric": "max", "metric_column": "annual_salary", "group_by": ["department"]}),
    Case("hr", "How many people did we hire each year?",
         lambda d: d.groupby(d.hire_date.dt.strftime("%Y-01-01")).size(), "groups",
         {"metric": "count", "time_grain": "year", "date_column": "hire_date", "limit": 100}),
    Case("hr", "Average salary for remote versus office employees",
         lambda d: d.groupby("is_remote").annual_salary.mean(), "groups",
         {"metric": "avg", "metric_column": "annual_salary", "group_by": ["is_remote"]}),
    # --- Roman Urdu ----------------------------------------------------------------------
    Case("sales", "Har region ka total revenue batao",
         lambda d: d.groupby("region").revenue.sum(), "groups",
         {**SUM_REV, "group_by": ["region"]}),
    Case("sales", "2025 mein kitne orders aaye?",
         lambda d: len(_between(d, "order_date", "2025-01-01", "2026-01-01")), "scalar",
         {"metric": "count", "date_column": "order_date", "date_phrase": "2025"}),
    Case("hr", "Engineering department mein average salary kitni hai?",
         lambda d: d[d.department == "Engineering"].annual_salary.mean(), "scalar",
         {"metric": "avg", "metric_column": "annual_salary",
          "filters": [{"column": "department", "op": "=", "value": "Engineering"}]}),
    Case("hr", "Har department mein kitne log hain?",
         lambda d: d.groupby("department").size(), "groups",
         {"metric": "count", "group_by": ["department"]}),
]  # fmt: skip


def _key(v: Any) -> str:
    return str(v).strip().lower()


def matches(expected: Any, rows: list[list[Any]], compare: Compare) -> bool:
    """Does a computed result (rows of [dims..., value]) give the expected answer?"""
    if not rows:
        return False
    if compare == "scalar":
        value = rows[0][-1] if len(rows) == 1 else None
        return value is not None and abs(float(value) - float(expected)) <= 1e-6 * max(
            1.0, abs(float(expected))
        )
    series: pd.Series = expected
    want = {
        (tuple(_key(x) for x in k) if isinstance(k, tuple) else (_key(k),)): float(v)
        for k, v in series.items()
    }
    got = {tuple(_key(x) for x in r[:-1]): r[-1] for r in rows}
    if compare == "first_key":
        return next(iter(want)) == tuple(_key(x) for x in rows[0][:-1])
    if compare == "ordered_keys":
        return [tuple(_key(x) for x in r[:-1]) for r in rows] == list(want)
    return set(got) == set(want) and all(
        abs(float(got[k]) - want[k]) <= 1e-6 * max(1.0, abs(want[k])) for k in want
    )
