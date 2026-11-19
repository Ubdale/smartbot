"""Executor results must equal an independent pandas computation over the same data."""

from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from app.connectors.csv_xlsx import ingest_file
from app.query.executor import (
    PlanError,
    ValidatedPlan,
    clear_cache,
    columns_from_profile,
    execute_plan,
    validate_plan,
)
from app.schemas.query_plan import QueryPlan

SAMPLE = Path(__file__).resolve().parent.parent / "sample_data"
TODAY = date(2026, 11, 19)


@pytest.fixture(scope="module")
def sales(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[dict[str, Any]]]:
    out = tmp_path_factory.mktemp("sales") / "data.parquet"
    res = ingest_file(SAMPLE / "sales_orders.csv", "csv", out)
    return out, [c.to_dict() for c in res.profile.columns]


@pytest.fixture(scope="module")
def hr(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[dict[str, Any]]]:
    out = tmp_path_factory.mktemp("hr") / "data.parquet"
    res = ingest_file(SAMPLE / "hr_employees.csv", "csv", out)
    return out, [c.to_dict() for c in res.profile.columns]


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    frame = pd.read_csv(SAMPLE / "sales_orders.csv", parse_dates=["order_date"])
    return frame


@pytest.fixture(autouse=True)
def _fresh_cache() -> Iterator[None]:
    clear_cache()
    yield
    clear_cache()


def run(
    data: tuple[Path, list[dict[str, Any]]], plan: dict[str, Any], **kw: Any
) -> tuple[ValidatedPlan, Any]:
    path, profile = data
    vp = validate_plan(QueryPlan.model_validate(plan), columns_from_profile(profile), TODAY)
    return vp, execute_plan(vp, path, dataset_id=kw.get("dataset_id", "ds"), version=kw.get("v", 1))


def as_dict(rows: list[list[Any]]) -> dict[Any, Any]:
    return {tuple(r[:-1]) if len(r) > 2 else r[0]: r[-1] for r in rows}


# --- correctness against pandas ------------------------------------------------------


def test_sum_by_group_sorted(sales: Any, df: pd.DataFrame) -> None:
    _, r = run(sales, {"metric": "sum", "metric_column": "revenue", "group_by": ["region"]})
    expected = df.groupby("region")["revenue"].sum().sort_values(ascending=False)
    assert [row[0] for row in r.rows] == list(expected.index)
    for row in r.rows:
        assert row[1] == pytest.approx(expected[row[0]])
    assert r.columns == ["region", "sum_revenue"]
    assert not r.truncated


def test_count_last_quarter(sales: Any, df: pd.DataFrame) -> None:
    vp, r = run(sales, {"metric": "count", "date_phrase": "last quarter"})
    mask = (df.order_date >= "2026-07-01") & (df.order_date < "2026-10-01")
    assert r.rows == [[int(mask.sum())]]
    assert vp.plan.date_column == "order_date"  # inferred: the only date column


def test_monthly_trend_with_case_insensitive_filter(sales: Any, df: pd.DataFrame) -> None:
    _, r = run(
        sales,
        {
            "metric": "sum",
            "metric_column": "revenue",
            "time_grain": "month",
            "date_column": "order_date",
            "date_phrase": "2025",
            "filters": [{"column": "category", "op": "=", "value": "subscriptions"}],
            "limit": 100,
        },
    )
    sub = df[(df.category == "Subscriptions") & (df.order_date.dt.year == 2025)]
    expected = sub.groupby(sub.order_date.dt.to_period("M"))["revenue"].sum()
    assert len(r.rows) == 12
    assert r.rows[0][0] == "2025-01-01"  # ordered by period ascending
    for (period, value), row in zip(expected.items(), r.rows, strict=True):
        assert row[0] == f"{period}-01"
        assert row[1] == pytest.approx(value)


def test_top_n_truncates(sales: Any, df: pd.DataFrame) -> None:
    _, r = run(
        sales,
        {
            "metric": "sum",
            "metric_column": "revenue",
            "group_by": ["customer"],
            "sort": {"by": "value", "direction": "desc"},
            "limit": 10,
        },
    )
    expected = df.groupby("customer")["revenue"].sum().sort_values(ascending=False).head(10)
    assert [row[0] for row in r.rows] == list(expected.index)
    assert r.truncated  # 45 customers > 10


def test_count_distinct_and_two_groups(sales: Any, df: pd.DataFrame) -> None:
    _, r = run(sales, {"metric": "count_distinct", "metric_column": "customer"})
    assert r.rows == [[df.customer.nunique()]]

    _, r = run(
        sales,
        {"metric": "avg", "metric_column": "quantity", "group_by": ["region", "category"],
         "limit": 100},
    )  # fmt: skip
    expected = df.groupby(["region", "category"])["quantity"].mean()
    got = as_dict(r.rows)
    assert len(got) == len(expected)
    for key, value in expected.items():
        assert got[key] == pytest.approx(value)


@pytest.mark.parametrize(
    ("flt", "mask_fn"),
    [
        ({"column": "region", "op": "in", "value": ["north", "ASIA"]},
         lambda d: d.region.isin(["North", "Asia"])),
        ({"column": "product", "op": "contains", "value": "plan"},
         lambda d: d["product"].str.contains("plan", case=False)),
        ({"column": "revenue", "op": "between", "value": [100, 500]},
         lambda d: d.revenue.between(100, 500)),
        ({"column": "quantity", "op": ">", "value": "20"}, lambda d: d.quantity > 20),
        ({"column": "region", "op": "!=", "value": "Europe"}, lambda d: d.region != "Europe"),
        ({"column": "order_date", "op": ">=", "value": "2026-01-01"},
         lambda d: d.order_date >= "2026-01-01"),
        ({"column": "discount", "op": "=", "value": 0}, lambda d: d.discount == 0),
    ],
)  # fmt: skip
def test_filters(sales: Any, df: pd.DataFrame, flt: dict[str, Any], mask_fn: Any) -> None:
    _, r = run(sales, {"metric": "count", "filters": [flt]})
    assert r.rows == [[int(mask_fn(df).sum())]]


def test_null_filter(hr: Any) -> None:
    hr_df = pd.read_csv(SAMPLE / "hr_employees.csv")
    _, r = run(hr, {"metric": "count", "filters": [{"column": "termination_date", "op": "="}]})
    assert r.rows == [[int(hr_df.termination_date.isna().sum())]]  # still employed


# --- safety -------------------------------------------------------------------------


def test_injection_in_values_is_just_a_value(sales: Any) -> None:
    evil = "x'); DROP TABLE t; SELECT ('1"
    _, r = run(
        sales, {"metric": "count", "filters": [{"column": "region", "op": "=", "value": evil}]}
    )
    assert r.rows == [[0]]
    assert evil not in r.sql
    assert evil in r.params


def test_sql_never_contains_values_or_server_paths(sales: Any) -> None:
    _, r = run(
        sales,
        {"metric": "sum", "metric_column": "revenue", "date_phrase": "last year",
         "filters": [{"column": "customer", "op": "contains", "value": "Apex"}]},
    )  # fmt: skip
    assert "Apex" not in r.sql and "2025-01-01" not in r.sql
    assert str(sales[0]) not in r.sql
    assert not any(str(sales[0]) in str(p) for p in r.params)


@pytest.mark.parametrize(
    ("plan", "message"),
    [
        ({"metric": "sum", "metric_column": "revenu"}, "Did you mean: 'revenue'"),
        ({"metric": "count", "group_by": ['region"; DROP TABLE t; --']}, "Unknown column"),
        ({"metric": "sum", "metric_column": "region"}, "not a number"),
        ({"metric": "sum"}, "needs a metric_column"),
        ({"metric": "count", "date_phrase": "last fortnight"}, "Couldn't understand"),
        ({"metric": "count", "date_column": "region"}, "not a date"),
        ({"metric": "count", "filters": [{"column": "revenue", "op": "contains", "value": "1"}]},
         "only works on text"),
        ({"metric": "count",
          "filters": [{"column": "region", "op": "between", "value": ["a", "b"]}]},
         "number or date"),
        ({"metric": "count", "filters": [{"column": "revenue", "op": ">", "value": "lots"}]},
         "doesn't fit"),
        ({"metric": "count", "filters": [{"column": "region", "op": ">"}]}, "needs a value"),
        ({"metric": "count", "group_by": ["region", "city"], "time_grain": "month"},
         "at most one"),
        ({"metric": "count", "group_by": ["region"], "sort": {"by": "nope"}}, "sort.by"),
    ],
)  # fmt: skip
def test_invalid_plans(sales: Any, plan: dict[str, Any], message: str) -> None:
    with pytest.raises(PlanError, match=message.replace("(", r"\(").replace(".", r"\.")):
        run(sales, plan)


def test_ambiguous_date_column_must_be_named(hr: Any) -> None:
    with pytest.raises(PlanError, match="hire_date"):
        run(hr, {"metric": "count", "date_phrase": "this year"})
    _, r = run(hr, {"metric": "count", "date_column": "hire_date", "date_phrase": "2025"})
    assert r.rows[0][0] > 0


def test_column_names_are_case_insensitive(sales: Any) -> None:
    vp, _ = run(sales, {"metric": "sum", "metric_column": "Revenue", "group_by": ["REGION"]})
    assert vp.plan.metric_column == "revenue"
    assert vp.plan.group_by == ["region"]


# --- cache --------------------------------------------------------------------------


def test_cache_hits_and_version_invalidation(sales: Any) -> None:
    plan = {"metric": "sum", "metric_column": "revenue", "group_by": ["region"]}
    _, first = run(sales, plan)
    _, second = run(sales, plan)
    _, other_version = run(sales, plan, v=2)
    assert not first.cached
    assert second.cached and second.rows == first.rows
    assert not other_version.cached


@pytest.mark.parametrize("by", ["revenue", "sum_revenue", "Value", "sum"])
def test_sort_by_metric_name_means_value(sales: Any, by: str) -> None:
    vp, r = run(
        sales,
        {"metric": "sum", "metric_column": "revenue", "group_by": ["region"],
         "sort": {"by": by, "direction": "asc"}},
    )  # fmt: skip
    assert vp.plan.sort is not None
    assert vp.plan.sort.by == "value"
    values = [row[1] for row in r.rows]
    assert values == sorted(values)


def test_time_grain_drops_the_raw_date_from_groups(hr: Any) -> None:
    vp, r = run(
        hr,
        {"metric": "count", "group_by": ["hire_date"], "time_grain": "year",
         "date_column": "hire_date", "limit": 100},
    )  # fmt: skip
    assert vp.plan.group_by == []
    assert r.columns == ["year", "count"]
