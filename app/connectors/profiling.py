"""Describe a Parquet dataset: column types, kinds, samples and simple stats.

This profile is what the LLM sees later (never the rows themselves).
"""

import math
import re
from dataclasses import asdict, dataclass
from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

import duckdb

from app.query.sql import quote_ident, quote_literal

ColumnKind = Literal["number", "text", "date", "datetime", "boolean"]

SAMPLE_VALUES = 3
MAX_SAMPLE_CHARS = 80

_NUMBER = re.compile(r"^(TINYINT|SMALLINT|INTEGER|BIGINT|HUGEINT|U\w*INT|FLOAT|DOUBLE|DECIMAL)")


def kind_of(duckdb_type: str) -> ColumnKind:
    t = duckdb_type.upper()
    if t == "BOOLEAN":
        return "boolean"
    if _NUMBER.match(t):
        return "number"
    if t == "DATE":
        return "date"
    if t.startswith("TIMESTAMP"):
        return "datetime"
    return "text"


def jsonable(value: Any) -> Any:
    """Convert a DuckDB result value into something JSON can carry without loss."""
    if value is None:
        return None
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, Decimal):
        # JSON numbers are doubles anyway; keep huge/precise values exact as strings.
        as_float = float(value)
        return as_float if abs(as_float) < 1e15 else str(value)
    if isinstance(value, datetime | date | time):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, bool | int | str):
        return value
    return str(value)


@dataclass(frozen=True)
class ColumnProfile:
    name: str
    type: str
    kind: ColumnKind
    null_count: int
    distinct_count: int  # approximate
    samples: list[Any]
    min: Any = None
    max: Any = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetProfile:
    row_count: int
    columns: list[ColumnProfile]


def _short(value: Any) -> Any:
    v = jsonable(value)
    if isinstance(v, str) and len(v) > MAX_SAMPLE_CHARS:
        return v[: MAX_SAMPLE_CHARS - 1] + "…"
    return v


def profile_parquet(path: Path) -> DatasetProfile:
    with duckdb.connect() as con:
        # Views can't take parameters; the path is server-generated and quoted.
        src = quote_literal(str(path))
        con.execute(f"CREATE VIEW t AS SELECT * FROM read_parquet({src})")  # noqa: S608
        described = con.execute("DESCRIBE t").fetchall()  # (name, type, null, key, default, extra)
        cols = [(str(r[0]), str(r[1])) for r in described]

        aggregates = ["count(*)"]
        for name, _ in cols:
            q = quote_ident(name)
            aggregates += [f"count({q})", f"approx_count_distinct({q})"]
        stats = con.execute(f"SELECT {', '.join(aggregates)} FROM t").fetchone()  # noqa: S608
        if stats is None:
            raise RuntimeError("aggregate query returned no row")
        row_count = int(stats[0])

        profiles: list[ColumnProfile] = []
        for i, (name, dtype) in enumerate(cols):
            q = quote_ident(name)
            kind = kind_of(dtype)
            non_null, distinct = int(stats[1 + 2 * i]), int(stats[2 + 2 * i])
            samples = [
                _short(r[0])
                for r in con.execute(
                    f"SELECT DISTINCT {q} FROM t WHERE {q} IS NOT NULL LIMIT ?",  # noqa: S608
                    [SAMPLE_VALUES],
                ).fetchall()
            ]
            lo = hi = None
            if kind in ("number", "date", "datetime"):
                bounds = con.execute(f"SELECT min({q}), max({q}) FROM t").fetchone()  # noqa: S608
                if bounds is not None:
                    lo, hi = jsonable(bounds[0]), jsonable(bounds[1])
            profiles.append(
                ColumnProfile(
                    name=name,
                    type=dtype,
                    kind=kind,
                    null_count=row_count - non_null,
                    distinct_count=min(distinct, non_null),
                    samples=samples,
                    min=lo,
                    max=hi,
                )
            )
    return DatasetProfile(row_count=row_count, columns=profiles)


def preview_parquet(path: Path, limit: int) -> tuple[list[str], list[list[Any]]]:
    with duckdb.connect() as con:
        rel = con.execute("SELECT * FROM read_parquet(?) LIMIT ?", [str(path), limit])
        columns = [d[0] for d in rel.description or []]
        rows = [[jsonable(v) for v in row] for row in rel.fetchall()]
    return columns, rows
