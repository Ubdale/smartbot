from datetime import date
from typing import Any

import pytest

from app.query.executor import ColumnInfo, QueryResult, validate_plan
from app.query.presentation import choose_chart, describe_plan, summary_is_grounded
from app.schemas.query_plan import QueryPlan

COLS = [
    ColumnInfo("order_date", "date"),
    ColumnInfo("region", "text"),
    ColumnInfo("revenue", "number"),
]


def vp(**plan: Any) -> Any:
    return validate_plan(QueryPlan.model_validate(plan), COLS, date(2026, 11, 19))


def result(rows: list[list[Any]], dims: list[str], value: str = "sum_revenue") -> QueryResult:
    return QueryResult([*dims, value], rows, False, "", [], dims, value)


REGION_ROWS = [["North", 1234567.89], ["Europe", 830000.5], ["Asia", 41250.0]]


@pytest.mark.parametrize(
    "summary",
    [
        "North leads with 1,234,567.89 in revenue, followed by Europe at 830,000.5.",
        "North brought in about 1.2 million, well ahead of Asia's 41,250.",
        "North is first at $1.23M and Asia is last with 41.3k.",
        "Revenue was highest in North across the 3 regions.",
        "Showing the top 5 regions as asked.",
    ],
)
def test_grounded_summaries_pass(summary: str) -> None:
    r = result(REGION_ROWS, ["region"])
    assert summary_is_grounded(summary, r, "top 5 regions by revenue", extra=[])


@pytest.mark.parametrize(
    "summary",
    [
        "North made 2,000,000 in revenue.",  # invented
        "Total revenue across regions was 2,105,818.39.",  # LLM did its own sum
        "North is 49% of revenue.",  # computed share
        "North's revenue grew 12% since 2025-01-01.",  # invented % and date
    ],
)
def test_ungrounded_summaries_fail(summary: str) -> None:
    r = result(REGION_ROWS, ["region"])
    assert not summary_is_grounded(summary, r, "revenue by region", extra=[])


def test_percent_of_fraction_is_allowed() -> None:
    r = result([["0.15", 120]], ["discount"], "count")
    r.rows = [[0.15, 120]]
    assert summary_is_grounded("120 orders had a 15% discount.", r, "q", extra=[])


def test_dates_must_come_from_result_or_range() -> None:
    r = result([["2026-07-01", 10.0], ["2026-08-01", 20.0]], ["month"])
    ok = "Revenue went from 10 in 2026-07-01 to 20 in 2026-08-01."
    assert summary_is_grounded(ok, r, "q", extra=[])
    assert summary_is_grounded("Covers 2026-07-01 to 2026-09-30.", r, "q", extra=["2026-09-30"])
    assert not summary_is_grounded("Peak was on 2026-07-15.", r, "q", extra=[])


def test_describe_plan() -> None:
    text = describe_plan(
        vp(
            metric="sum",
            metric_column="revenue",
            group_by=["region"],
            date_phrase="last quarter",
            filters=[{"column": "region", "op": "in", "value": ["North", "Asia"]}],
            sort={"by": "value", "direction": "desc"},
            limit=5,
        )
    )
    assert text == (
        "Total revenue, by region, where order_date from 2026-07-01 to 2026-09-30 "
        "(last quarter) and region is one of North, Asia; sorted by value, highest first; "
        "showing up to 5 rows."
    )


def test_chart_rules() -> None:
    by_region = result(REGION_ROWS, ["region"])
    assert choose_chart(vp(metric="sum", metric_column="revenue", group_by=["region"],
                           chart="pie"), by_region).type == "pie"  # type: ignore[union-attr]  # fmt: skip
    # A line chart without a time axis falls back to bar.
    spec = choose_chart(
        vp(metric="sum", metric_column="revenue", group_by=["region"], chart="line"), by_region
    )
    assert spec is not None and spec.type == "bar"
    # Trends get a line chart even if the model said "none".
    trend = result([["2026-01-01", 1.0], ["2026-02-01", 2.0]], ["month"])
    spec = choose_chart(vp(metric="sum", metric_column="revenue", time_grain="month"), trend)
    assert spec is not None and (spec.type, spec.x) == ("line", "month")
    # Single numbers never get a chart.
    single = result([[42.0]], [])
    assert choose_chart(vp(metric="sum", metric_column="revenue", chart="bar"), single) is None
    # avg can't be a pie (shares of a total make no sense).
    spec = choose_chart(
        vp(metric="avg", metric_column="revenue", group_by=["region"], chart="pie"), by_region
    )
    assert spec is not None and spec.type == "bar"
