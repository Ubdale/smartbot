"""Turn a time phrase ("last quarter", "Q3 2025", "since 2025-03") into exact dates.

Small LLMs are bad at calendar arithmetic, so the model only copies the phrase
and this module resolves it. Ranges are half-open: start <= d < end.
"""

import calendar
import re
from dataclasses import dataclass
from datetime import date, timedelta


class DatePhraseError(ValueError):
    pass


@dataclass(frozen=True)
class DateRange:
    start: date
    end: date  # exclusive
    label: str

    @property
    def last_day(self) -> date:
        return self.end - timedelta(days=1)


MONTHS = {
    name.lower(): i
    for i in range(1, 13)
    for name in (calendar.month_name[i], calendar.month_abbr[i])
}
MONTHS["sept"] = 9

_UNITS = {
    "day": "day",
    "days": "day",
    "week": "week",
    "weeks": "week",
    "month": "month",
    "months": "month",
    "quarter": "quarter",
    "quarters": "quarter",
    "year": "year",
    "years": "year",
}
_WORD_NUMBERS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
                 "eight": 8, "nine": 9, "ten": 10, "twelve": 12}  # fmt: skip
_ALL_TIME = {"all time", "all-time", "ever", "overall", "all", "any time", "total"}


def add_months(d: date, months: int) -> date:
    y, m = divmod(d.month - 1 + months, 12)
    year, month = d.year + y, m + 1
    return date(year, month, min(d.day, calendar.monthrange(year, month)[1]))


def _month(year: int, month: int) -> tuple[date, date]:
    start = date(year, month, 1)
    return start, add_months(start, 1)


def _quarter(year: int, q: int) -> tuple[date, date]:
    start = date(year, 3 * (q - 1) + 1, 1)
    return start, add_months(start, 3)


def _year(year: int) -> tuple[date, date]:
    return date(year, 1, 1), date(year + 1, 1, 1)


def _week_start(d: date) -> date:
    return d - timedelta(days=d.weekday())  # Monday


def _period(unit: str, d: date) -> tuple[date, date]:
    """The calendar period of `unit` containing `d`."""
    if unit == "day":
        return d, d + timedelta(days=1)
    if unit == "week":
        s = _week_start(d)
        return s, s + timedelta(days=7)
    if unit == "month":
        return _month(d.year, d.month)
    if unit == "quarter":
        return _quarter(d.year, (d.month - 1) // 3 + 1)
    return _year(d.year)


def _shift(unit: str, d: date, n: int) -> date:
    if unit == "day":
        return d + timedelta(days=n)
    if unit == "week":
        return d + timedelta(weeks=n)
    if unit == "month":
        return add_months(d, n)
    if unit == "quarter":
        return add_months(d, 3 * n)
    return add_months(d, 12 * n)


def _point(text: str, today: date) -> tuple[date, date]:
    """A single named period: ISO date, YYYY-MM, 'March 2025', 'Q3 2025', '2025'."""
    t = text.strip()
    if m := re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", t):
        d = date(int(m[1]), int(m[2]), int(m[3]))
        return d, d + timedelta(days=1)
    if m := re.fullmatch(r"(\d{4})-(\d{1,2})", t):
        return _month(int(m[1]), int(m[2]))
    if m := re.fullmatch(r"(\d{4})", t):
        return _year(int(m[1]))
    if m := re.fullmatch(r"q([1-4])\s*(?:of\s*)?(\d{4})|(\d{4})\s*-?\s*q([1-4])", t):
        q, y = (int(m[1]), int(m[2])) if m[1] else (int(m[4]), int(m[3]))
        return _quarter(y, q)
    if m := re.fullmatch(r"q([1-4])", t):
        q = int(m[1])
        start, end = _quarter(today.year, q)
        return (start, end) if start <= today else _quarter(today.year - 1, q)
    if m := re.fullmatch(r"h([12])\s*(?:of\s*)?(\d{4})", t):
        start = date(int(m[2]), 1 if m[1] == "1" else 7, 1)
        return start, add_months(start, 6)
    if (m := re.fullmatch(r"([a-z]+)\.?\s*,?\s*(\d{4})", t)) and m[1] in MONTHS:
        return _month(int(m[2]), MONTHS[m[1]])
    if t in MONTHS:  # most recent such month that has started
        start, end = _month(today.year, MONTHS[t])
        return (start, end) if start <= today else _month(today.year - 1, MONTHS[t])
    raise DatePhraseError(text)


def resolve_date_phrase(phrase: str, today: date) -> DateRange | None:
    """Resolve a time phrase. Returns None for "all time"; raises DatePhraseError if unknown."""
    p = re.sub(r"\s+", " ", phrase.lower().strip().strip("."))
    p = p.replace(chr(0x2019), "'")  # curly apostrophe from phones/Word
    p = re.sub(r"^(in|during|for|over|within|of|from)\s+(the\s+)?", "", p)
    p = re.sub(r"^the\s+", "", p)
    if p in _ALL_TIME:
        return None

    def rng(start: date, end: date) -> DateRange:
        return DateRange(start, end, phrase.strip())

    if p == "today":
        return rng(*_period("day", today))
    if p == "yesterday":
        return rng(*_period("day", today - timedelta(days=1)))

    to_date = {"ytd": "year", "mtd": "month", "qtd": "quarter", "wtd": "week"}
    if p in to_date:
        return rng(_period(to_date[p], today)[0], today + timedelta(days=1))
    if m := re.fullmatch(r"(year|month|quarter|week)[- ]to[- ]date", p):
        return rng(_period(m[1], today)[0], today + timedelta(days=1))
    if m := re.fullmatch(r"(?:this|current) (year|month|quarter|week)(?: so far)?", p):
        return rng(*_period(m[1], today))
    if m := re.fullmatch(r"(?:last|previous|prior|past) (year|month|quarter|week)", p):
        return rng(*_period(m[1], _shift(m[1], today, -1)))
    if m := re.fullmatch(r"(?:last|past|previous|trailing) (\d+|[a-z]+) ([a-z]+)", p):
        n = int(m[1]) if m[1].isdigit() else _WORD_NUMBERS.get(m[1], 0)
        unit = _UNITS.get(m[2])
        if n > 0 and unit:
            # Rolling window ending today (inclusive), e.g. last 7 days = today and 6 before.
            end = today + timedelta(days=1)
            start = end - timedelta(days=n) if unit == "day" else _shift(unit, end, -n)
            return rng(start, end)
    if m := re.fullmatch(r"(?:since|after|starting|from) (.+)", p):
        return rng(_point(m[1], today)[0], today + timedelta(days=1))
    if m := re.fullmatch(r"before (.+)", p):
        return rng(date(1900, 1, 1), _point(m[1], today)[0])
    if m := re.fullmatch(r"(?:between )?(.+?) (?:to|and|until|through|-) (.+)", p):
        start = _point(m[1], today)[0]
        end = _point(m[2], today)[1]
        if end <= start:
            raise DatePhraseError(phrase)
        return rng(start, end)
    try:
        return rng(*_point(p, today))
    except (DatePhraseError, ValueError):
        raise DatePhraseError(phrase) from None
