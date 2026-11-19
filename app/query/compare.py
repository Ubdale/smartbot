"""What changed between two versions of a dataset (e.g. this week's Jira export vs last
week's): added rows, removed rows, and changed values, matched by an ID column.

Everything is computed by DuckDB over the two Parquet files; nothing is estimated.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb

from app.connectors.profiling import jsonable
from app.query.sql import quote_ident, quote_literal

MAX_CHANGES = 500  # cell changes listed (counts always cover everything)
_KEY_NAME = re.compile(r"(^|[\s_\-])(id|key|code|no|number|ref|sku|uuid)($|[\s_\-])", re.I)


@dataclass
class Comparison:
    key: str | None  # the column rows were matched by (None: whole rows compared)
    rows_before: int
    rows_after: int
    added: int
    removed: int
    changed_rows: int
    changes_by_column: dict[str, int] = field(default_factory=dict)
    # The most common value changes per column, e.g. Status: ("Open", "Done") x 7.
    transitions: dict[str, list[tuple[Any, Any, int]]] = field(default_factory=dict)
    added_columns: list[str] = field(default_factory=list)
    removed_columns: list[str] = field(default_factory=list)
    # [change, key, column, before, after]; at most MAX_CHANGES
    rows: list[list[Any]] = field(default_factory=list)
    truncated: bool = False

    @property
    def nothing(self) -> bool:
        return not (self.added or self.removed or self.changed_rows or self.added_columns
                    or self.removed_columns)  # fmt: skip


def _columns(con: duckdb.DuckDBPyConnection, table: str) -> dict[str, str]:
    return {r[0]: r[1] for r in con.execute(f"DESCRIBE {table}").fetchall()}


def _is_unique_key(con: duckdb.DuckDBPyConnection, table: str, col: str) -> bool:
    q = quote_ident(col)
    total, distinct, nulls = con.execute(
        f"SELECT count(*), count(DISTINCT {q}), count(*) - count({q}) FROM {table}"  # noqa: S608
    ).fetchone() or (0, 0, 1)
    return bool(total > 0 and nulls == 0 and distinct == total)


def find_key(con: duckdb.DuckDBPyConnection, common: list[str]) -> str | None:
    """A column that identifies rows in both versions: unique and never empty.
    Prefers ID-like names ("Issue key", "order_id"), then the leftmost such column."""
    candidates = [c for c in common if _is_unique_key(con, "before", c)
                  and _is_unique_key(con, "after", c)]  # fmt: skip
    named = [c for c in candidates if _KEY_NAME.search(c)]
    best = named or candidates
    return best[0] if best else None


def compare_versions(before: Path, after: Path) -> Comparison:
    with duckdb.connect() as con:
        con.execute("SET threads TO 2")
        # Server-generated paths: views can't take parameters, so quote them as literals.
        for view, path in (("before", before), ("after", after)):
            source = f"read_parquet({quote_literal(str(path))})"
            con.execute(f"CREATE VIEW {view} AS SELECT * FROM {source}")  # noqa: S608
        cols_before, cols_after = _columns(con, "before"), _columns(con, "after")
        common = [c for c in cols_after if c in cols_before]
        rows_before = int((con.execute("SELECT count(*) FROM before").fetchone() or [0])[0])
        rows_after = int((con.execute("SELECT count(*) FROM after").fetchone() or [0])[0])
        result = Comparison(
            key=None, rows_before=rows_before, rows_after=rows_after, added=0, removed=0,
            changed_rows=0,
            added_columns=[c for c in cols_after if c not in cols_before],
            removed_columns=[c for c in cols_before if c not in cols_after],
        )  # fmt: skip
        if not common:
            return result
        key = find_key(con, common)
        result.key = key
        if key is None:
            _compare_whole_rows(con, common, result)
        else:
            _compare_by_key(con, key, common, cols_before, cols_after, result)
        return result


def _compare_whole_rows(
    con: duckdb.DuckDBPyConnection, common: list[str], result: Comparison
) -> None:
    """No ID column: rows that appear only before (removed) or only after (added)."""
    cols = ", ".join(quote_ident(c) for c in common)
    for label, sql in (
        ("Added", f"SELECT {cols} FROM after EXCEPT ALL SELECT {cols} FROM before"),  # noqa: S608
        ("Removed", f"SELECT {cols} FROM before EXCEPT ALL SELECT {cols} FROM after"),  # noqa: S608
    ):
        count = int((con.execute(f"SELECT count(*) FROM ({sql})").fetchone() or [0])[0])  # noqa: S608
        if label == "Added":
            result.added = count
        else:
            result.removed = count
        for row in con.execute(f"{sql} LIMIT {MAX_CHANGES}").fetchall():
            if len(result.rows) >= MAX_CHANGES:
                result.truncated = True
                break
            summary = ", ".join(f"{c}: {jsonable(v)}" for c, v in zip(common, row, strict=True)
                                if v is not None)  # fmt: skip
            result.rows.append([label, None, "", "", summary[:300]])
    result.truncated = result.truncated or result.added + result.removed > len(result.rows)


def _compare_by_key(
    con: duckdb.DuckDBPyConnection,
    key: str,
    common: list[str],
    cols_before: dict[str, str],
    cols_after: dict[str, str],
    result: Comparison,
) -> None:
    k = quote_ident(key)
    kb, ka = f"CAST(b.{k} AS VARCHAR)", f"CAST(a.{k} AS VARCHAR)"
    join = f"FROM before b JOIN after a ON {kb} = {ka}"
    added_sql = f"SELECT {ka} FROM after a WHERE {ka} NOT IN (SELECT {kb} FROM before b)"  # noqa: S608
    removed_sql = f"SELECT {kb} FROM before b WHERE {kb} NOT IN (SELECT {ka} FROM after a)"  # noqa: S608
    result.added = int((con.execute(f"SELECT count(*) FROM ({added_sql})").fetchone() or [0])[0])  # noqa: S608
    result.removed = int(
        (con.execute(f"SELECT count(*) FROM ({removed_sql})").fetchone() or [0])[0]  # noqa: S608
    )

    # One row per changed cell, for every column both versions have.
    parts = []
    for c in common:
        if c == key:
            continue
        q = quote_ident(c)
        same_type = cols_before[c] == cols_after[c]
        b, a = (f"b.{q}", f"a.{q}") if same_type else (f"CAST(b.{q} AS VARCHAR)",
                                                        f"CAST(a.{q} AS VARCHAR)")  # fmt: skip
        parts.append(
            f"SELECT {kb} AS k, {quote_literal(c)} AS col, CAST({b} AS VARCHAR) AS before_value, "  # noqa: S608
            f"CAST({a} AS VARCHAR) AS after_value {join} WHERE {b} IS DISTINCT FROM {a}"
        )
    con.execute("CREATE TEMP TABLE cell_changes AS " + (" UNION ALL ".join(parts) if parts else
                "SELECT NULL::VARCHAR AS k, NULL::VARCHAR AS col, NULL::VARCHAR AS before_value, "
                "NULL::VARCHAR AS after_value WHERE false"))  # fmt: skip
    result.changed_rows = int(
        (con.execute("SELECT count(DISTINCT k) FROM cell_changes").fetchone() or [0])[0]
    )
    result.changes_by_column = {
        str(col): int(n)
        for col, n in con.execute(
            "SELECT col, count(*) FROM cell_changes GROUP BY col ORDER BY count(*) DESC, col"
        ).fetchall()
    }
    for col, before_v, after_v, n in con.execute(
        "SELECT col, before_value, after_value, count(*) AS n FROM cell_changes "
        "GROUP BY ALL ORDER BY n DESC, col"
    ).fetchall():
        bucket = result.transitions.setdefault(str(col), [])
        if len(bucket) < 5:
            bucket.append((before_v, after_v, int(n)))

    rows: list[list[Any]] = []
    for (k_value,) in con.execute(f"{added_sql} ORDER BY 1 LIMIT {MAX_CHANGES}").fetchall():
        rows.append(["Added", k_value, "", "", ""])
    for (k_value,) in con.execute(f"{removed_sql} ORDER BY 1 LIMIT {MAX_CHANGES}").fetchall():
        rows.append(["Removed", k_value, "", "", ""])
    for k_value, col, before_v, after_v in con.execute(
        f"SELECT * FROM cell_changes ORDER BY k, col LIMIT {MAX_CHANGES}"  # noqa: S608
    ).fetchall():
        rows.append(["Changed", k_value, col, before_v, after_v])
    total = result.added + result.removed + sum(result.changes_by_column.values())
    result.rows = rows[:MAX_CHANGES]
    result.truncated = total > len(result.rows)


def describe_comparison(c: Comparison, name: str) -> str:
    """A plain-English summary written by code (every number is computed above)."""
    if c.nothing:
        return f"Nothing changed in “{name}” since the previous version."
    parts = []
    if c.added:
        parts.append(f"{c.added:,} new row{'s' if c.added != 1 else ''}")
    if c.removed:
        parts.append(f"{c.removed:,} removed")
    if c.changed_rows:
        parts.append(f"{c.changed_rows:,} changed")
    text = (f"Since the previous version of “{name}” ({c.rows_before:,} → "
            f"{c.rows_after:,} rows): " + ", ".join(parts or ["no row changes"]) + ".")  # fmt: skip
    if c.key:
        text += f" Rows were matched by {c.key}."
    top = list(c.changes_by_column.items())[:3]
    if top:
        details = []
        for col, n in top:
            moves = c.transitions.get(col, [])
            common = moves[0] if moves and moves[0][2] > 1 else None
            if common:
                details.append(f"{col} ({n:,}; most often {common[0]} → {common[1]}, "
                               f"{common[2]:,} times)")  # fmt: skip
            else:
                details.append(f"{col} ({n:,})")
        text += " Most changes: " + "; ".join(details) + "."
    if c.added_columns:
        text += " New columns: " + ", ".join(c.added_columns[:5]) + "."
    if c.removed_columns:
        text += " Columns no longer present: " + ", ".join(c.removed_columns[:5]) + "."
    return text


def comparison_counts(c: Comparison) -> dict[str, Any]:
    return {"key": c.key, "added": c.added, "removed": c.removed,
            "changed": c.changed_rows, "rows_before": c.rows_before,
            "rows_after": c.rows_after}  # fmt: skip
