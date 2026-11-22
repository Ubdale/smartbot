"""Combine two datasets (e.g. two sheets of a workbook) into one answer, by joining on
a shared key column. The join runs in DuckDB; numbers are never invented."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
from rapidfuzz import fuzz

from app.connectors.profiling import jsonable
from app.exporting import is_id_like
from app.query.sql import quote_ident, quote_literal

MAX_ROWS = 300
MAX_COLUMNS = 40


class CombineError(ValueError):
    """Couldn't combine the two datasets (no shared column, etc.). Message is user-facing."""


@dataclass
class Combined:
    columns: list[str]
    rows: list[list[Any]]
    key: str
    left_name: str
    right_name: str
    matched: int  # rows of the left that found a match on the right
    total_left: int
    truncated: bool = False
    notes: list[str] = field(default_factory=list)


def _names(profile: list[dict[str, Any]]) -> list[str]:
    return [str(c["name"]) for c in profile]


def shared_columns(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> list[str]:
    """Columns present in both (matched case-insensitively; left spelling kept)."""
    right_lower = {n.lower(): n for n in _names(right)}
    return [n for n in _names(left) if n.lower() in right_lower]


def choose_key(left: list[dict[str, Any]], right: list[dict[str, Any]], wanted: str | None) -> str:
    """The column to join on: the one the user asked for, else the best shared column
    (an ID-like one first, otherwise the first shared column)."""
    shared = shared_columns(left, right)
    if wanted:
        match = next((c for c in shared if c.lower() == wanted.strip().lower()), None)
        if match is None:
            # Maybe the user named a column that exists in both but spelled differently.
            both = _names(left) + _names(right)
            near = max(both, key=lambda c: fuzz.ratio(c.lower(), wanted.lower()), default="")
            shared_text = (
                f"Shared columns: {', '.join(shared)}."
                if shared
                else "The two datasets share no column to join on."
            )
            raise CombineError(
                f"I couldn't find a column called “{wanted}” in both datasets. "
                + (f"Did you mean “{near}”? " if near else "")
                + shared_text
            )
        return match
    if not shared:
        raise CombineError(
            "These two datasets share no column to join on. Combining needs a common "
            "column (such as an ID) that links the rows."
        )
    ids = [c for c in shared if is_id_like(c)]
    return (ids or shared)[0]


def _right_column_name(col: str, left_names: set[str], key: str) -> str:
    """Rename a right-hand column that clashes with a left one (keep the key single)."""
    if col == key:
        return col
    return f"{col} (2)" if col in left_names else col


def combine(
    left_parquet: Path, right_parquet: Path, *, left_name: str, right_name: str,
    left_profile: list[dict[str, Any]], right_profile: list[dict[str, Any]], key: str,
) -> Combined:  # fmt: skip
    """Left-join the two datasets on `key` and return a capped preview."""
    left_names = _names(left_profile)
    right_names = _names(right_profile)
    right_key = next(n for n in right_names if n.lower() == key.lower())
    left_set = set(left_names)

    left_select = ", ".join(f"l.{quote_ident(c)}" for c in left_names)
    extra_right = [c for c in right_names if c.lower() != key.lower()]
    right_select = ", ".join(
        f"r.{quote_ident(c)} AS {quote_ident(_right_column_name(c, left_set, key))}"
        for c in extra_right
    )
    select = left_select + (", " + right_select if right_select else "")
    out_columns = left_names + [_right_column_name(c, left_set, key) for c in extra_right]
    out_columns = out_columns[:MAX_COLUMNS]

    lp, rp = quote_literal(str(left_parquet)), quote_literal(str(right_parquet))
    join_on = f"CAST(l.{quote_ident(key)} AS VARCHAR) = CAST(r.{quote_ident(right_key)} AS VARCHAR)"
    base = f"FROM read_parquet({lp}) l LEFT JOIN read_parquet({rp}) r ON {join_on}"
    with duckdb.connect() as con:
        con.execute("SET threads TO 2")
        total_left = int(
            (con.execute(f"SELECT count(*) FROM read_parquet({lp})").fetchone() or [0])[0]  # noqa: S608
        )
        matched = int(
            (
                con.execute(
                    f"SELECT count(*) {base} WHERE r.{quote_ident(right_key)} IS NOT NULL"
                ).fetchone()
                or [0]
            )[0]
        )
        cur = con.execute(f"SELECT {select} {base} LIMIT {MAX_ROWS + 1}")
        columns = [d[0] for d in cur.description or []][:MAX_COLUMNS]
        raw = cur.fetchmany(MAX_ROWS + 1)

    truncated = len(raw) > MAX_ROWS
    rows = [[jsonable(v) for v in r[: len(columns)]] for r in raw[:MAX_ROWS]]
    notes = []
    if matched < total_left:
        notes.append(
            f"{total_left - matched:,} of {total_left:,} rows in “{left_name}” had no "
            f"match in “{right_name}” (shown with blank extra columns)."
        )
    if len(out_columns) >= MAX_COLUMNS:
        notes.append(f"Showing the first {MAX_COLUMNS} columns.")
    return Combined(columns, rows, key, left_name, right_name, matched, total_left, truncated,
                    notes)  # fmt: skip


def describe_combined(c: Combined) -> str:
    text = (f"Combined “{c.left_name}” with “{c.right_name}” on {c.key}: "
            f"{c.matched:,} of {c.total_left:,} rows matched.")  # fmt: skip
    if c.notes:
        text += " " + " ".join(c.notes)
    return text
