from datetime import date

import pytest

from app.query.dates import DatePhraseError, resolve_date_phrase

TODAY = date(2026, 11, 19)  # a Thursday


@pytest.mark.parametrize(
    ("phrase", "start", "end"),
    [
        ("today", "2026-11-19", "2026-11-20"),
        ("yesterday", "2026-11-18", "2026-11-19"),
        ("this week", "2026-11-16", "2026-11-23"),
        ("last week", "2026-11-09", "2026-11-16"),
        ("this month", "2026-11-01", "2026-12-01"),
        ("last month", "2026-10-01", "2026-11-01"),
        ("previous month", "2026-10-01", "2026-11-01"),
        ("this quarter", "2026-10-01", "2027-01-01"),
        ("last quarter", "2026-07-01", "2026-10-01"),
        ("in the last quarter", "2026-07-01", "2026-10-01"),
        ("This Year", "2026-01-01", "2027-01-01"),
        ("last year", "2025-01-01", "2026-01-01"),
        ("YTD", "2026-01-01", "2026-11-20"),
        ("year to date", "2026-01-01", "2026-11-20"),
        ("month-to-date", "2026-11-01", "2026-11-20"),
        ("qtd", "2026-10-01", "2026-11-20"),
        ("last 7 days", "2026-11-13", "2026-11-20"),
        ("past 30 days", "2026-10-21", "2026-11-20"),
        ("last three months", "2026-08-20", "2026-11-20"),
        ("last 2 years", "2024-11-20", "2026-11-20"),
        ("2025", "2025-01-01", "2026-01-01"),
        ("in 2025", "2025-01-01", "2026-01-01"),
        ("Q3 2025", "2025-07-01", "2025-10-01"),
        ("2025-Q3", "2025-07-01", "2025-10-01"),
        ("q4", "2026-10-01", "2027-01-01"),
        ("H2 2025", "2025-07-01", "2026-01-01"),
        ("March 2025", "2025-03-01", "2025-04-01"),
        ("sep 2025", "2025-09-01", "2025-10-01"),
        ("march", "2026-03-01", "2026-04-01"),
        ("december", "2025-12-01", "2026-01-01"),  # Dec 2026 hasn't started yet
        ("2025-03", "2025-03-01", "2025-04-01"),
        ("2025-03-15", "2025-03-15", "2025-03-16"),
        ("since 2025-06-01", "2025-06-01", "2026-11-20"),
        ("since March 2026", "2026-03-01", "2026-11-20"),
        ("before 2025", "1900-01-01", "2025-01-01"),
        ("2025-01-01 to 2025-03-31", "2025-01-01", "2025-04-01"),
        ("from jan 2025 to mar 2025", "2025-01-01", "2025-04-01"),
        ("between 2024 and 2025", "2024-01-01", "2026-01-01"),
    ],
)
def test_resolves(phrase: str, start: str, end: str) -> None:
    r = resolve_date_phrase(phrase, TODAY)
    assert r is not None
    assert (r.start.isoformat(), r.end.isoformat()) == (start, end)
    assert r.label == phrase.strip()


@pytest.mark.parametrize("phrase", ["all time", "ever", "overall"])
def test_all_time_means_no_filter(phrase: str) -> None:
    assert resolve_date_phrase(phrase, TODAY) is None


@pytest.mark.parametrize(
    "phrase", ["last fortnight", "soon", "2025-13", "q5 2025", "2025-03-01 to 2024-01-01", ""]
)
def test_unknown_phrases_raise(phrase: str) -> None:
    with pytest.raises(DatePhraseError):
        resolve_date_phrase(phrase, TODAY)


def test_month_end_clamping() -> None:
    r = resolve_date_phrase("last month", date(2026, 3, 31))
    assert r is not None
    assert (r.start, r.end) == (date(2026, 2, 1), date(2026, 3, 1))
    r = resolve_date_phrase("last 1 months", date(2026, 3, 31))
    assert r is not None
    assert r.start == date(2026, 3, 1)
