"""Data doctor: finds problems in a table and fixes the safe ones.

Checks (all computed by code; no AI guessing): duplicate rows, duplicate IDs, empty
columns, blanks, extra spaces, the same value spelled several ways ("Sara Ahmed",
"sara ahmed ", "Sara  Ahmed"), numbers stored as text, dates in the future and extreme
values. Safe fixes (remove duplicate rows, tidy spaces, merge spellings, drop empty
columns) write a new version of the dataset, so "what changed?" shows exactly what
was fixed and the previous version is kept.
"""

import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Literal

import duckdb
from rapidfuzz import fuzz

from app.connectors.profiling import jsonable
from app.exporting import is_id_like
from app.query.sql import quote_ident, quote_literal

Severity = Literal["high", "medium", "low"]
FixKind = Literal["drop_duplicates", "trim", "merge", "drop_columns"]
MAX_LABEL_DISTINCT = 3000
MAX_BLANK_ISSUES = 8
_PLANNED = re.compile(r"due|deadline|end|expir|target|plan|schedul|forecast|next|renew", re.I)
_NUMBERISH = re.compile(r"^\s*[-+(]?\s*[$€£₨]?\s*[\d.,]+\s*%?\)?\s*$")


@dataclass
class Issue:
    id: str
    severity: Severity
    title: str
    column: str | None
    rows: int  # rows affected
    detail: str
    examples: list[str] = field(default_factory=list)
    fix: dict[str, Any] | None = None  # machine-applicable fix, or None
    fix_label: str | None = None  # what the fix does, for the button

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "severity": self.severity, "title": self.title,
                "column": self.column, "rows": self.rows, "detail": self.detail,
                "examples": self.examples, "fixable": self.fix is not None,
                "fix_label": self.fix_label}  # fmt: skip


def _norm(value: str) -> str:
    return re.sub(r"[^0-9a-z]+", " ", value.lower()).strip()


def _scalar(con: duckdb.DuckDBPyConnection, sql: str, params: list[Any] | None = None) -> Any:
    row = con.execute(sql, params or []).fetchone()
    return row[0] if row else None


def diagnose(parquet: Path, profile: list[dict[str, Any]], today: date) -> list[Issue]:
    issues: list[Issue] = []
    with duckdb.connect() as con:
        con.execute("SET threads TO 2")
        # A server-generated path: views can't take parameters, so it's quoted as a literal.
        con.execute(f"CREATE VIEW t AS SELECT * FROM read_parquet({quote_literal(str(parquet))})")  # noqa: S608
        total = int(_scalar(con, "SELECT count(*) FROM t") or 0)
        if total == 0:
            return [Issue("empty", "high", "The table has no rows", None, 0,
                          "There is nothing to analyse.")]  # fmt: skip
        kinds = {str(c["name"]): str(c["kind"]) for c in profile}

        # 1. Exact duplicate rows
        distinct = int(_scalar(con, "SELECT count(*) FROM (SELECT DISTINCT * FROM t)") or 0)
        if distinct < total:
            n = total - distinct
            issues.append(Issue(
                "duplicates", "high", "Duplicate rows", None, n,
                f"{n:,} row{'s are' if n != 1 else ' is'} an exact copy of another row. "
                "They are counted twice in totals.",
                fix={"type": "drop_duplicates"}, fix_label=f"Remove {n:,} duplicate rows",
            ))  # fmt: skip

        empty_columns = []
        blanks: list[tuple[str, int]] = []
        for name, kind in kinds.items():
            q = quote_ident(name)
            is_text = kind == "text"
            blank_sql = (f"{q} IS NULL OR trim(CAST({q} AS VARCHAR)) = ''" if is_text
                         else f"{q} IS NULL")  # fmt: skip
            n_blank = int(_scalar(con, f"SELECT count(*) FROM t WHERE {blank_sql}") or 0)  # noqa: S608
            if n_blank == total:
                empty_columns.append(name)
                continue
            if n_blank:
                blanks.append((name, n_blank))

            # 2. Duplicate IDs
            if is_id_like(name) and kind in ("text", "number"):
                dup = con.execute(
                    f"SELECT CAST({q} AS VARCHAR), count(*) FROM t WHERE {q} IS NOT NULL "  # noqa: S608
                    "GROUP BY 1 HAVING count(*) > 1 ORDER BY 2 DESC LIMIT 5"
                ).fetchall()
                if dup:
                    n_dup = int(_scalar(con, f"SELECT count(*) - count(DISTINCT {q}) FROM t "  # noqa: S608
                                             f"WHERE {q} IS NOT NULL") or 0)  # fmt: skip
                    issues.append(Issue(
                        f"dup-id:{name}", "high", "Repeated IDs", name, n_dup,
                        f"{name} should identify each row, but {n_dup:,} rows repeat an ID "
                        "used by another row (with different details).",
                        examples=[f"{v} ({c} times)" for v, c in dup],
                    ))  # fmt: skip

            if not is_text:
                continue
            # 3. Extra spaces
            n_space = int(_scalar(con, (
                f"SELECT count(*) FROM t WHERE {q} IS NOT NULL AND "  # noqa: S608
                f"{q} <> trim(regexp_replace({q}, '\\s+', ' ', 'g'))")) or 0)  # fmt: skip
            if n_space:
                ex = [str(r[0]) for r in con.execute(
                    f"SELECT DISTINCT {q} FROM t WHERE {q} <> "  # noqa: S608
                    f"trim(regexp_replace({q}, '\\s+', ' ', 'g')) LIMIT 3").fetchall()]  # fmt: skip
                issues.append(Issue(
                    f"spaces:{name}", "low", "Extra spaces", name, n_space,
                    f"{n_space:,} values in {name} start or end with spaces, or have double "
                    "spaces, so identical values can count as different ones.",
                    examples=[repr(e) for e in ex],
                    fix={"type": "trim", "column": name}, fix_label="Tidy the spaces",
                ))  # fmt: skip

            values = con.execute(
                f"SELECT CAST({q} AS VARCHAR) AS v, count(*) FROM t WHERE {q} IS NOT NULL "  # noqa: S608
                f"GROUP BY 1 LIMIT {MAX_LABEL_DISTINCT + 1}"
            ).fetchall()
            if len(values) > MAX_LABEL_DISTINCT:
                continue
            counts = {str(v): int(n) for v, n in values}

            # 4. Numbers stored as text
            non_blank = [v for v in counts if v.strip()]
            numeric = [v for v in non_blank if _NUMBERISH.match(v)]
            if non_blank and len(numeric) >= 0.9 * len(non_blank) and len(numeric) < len(non_blank):
                odd = [v for v in non_blank if not _NUMBERISH.match(v)]
                n_odd = sum(counts[v] for v in odd)
                issues.append(Issue(
                    f"numtext:{name}", "medium", "Numbers stored as text", name, n_odd,
                    f"{name} is almost all numbers, but {n_odd:,} values aren't, so it can't "
                    "be added up or averaged. Fix these values in the source file.",
                    examples=odd[:5],
                ))  # fmt: skip
                continue

            # 5. One value, several spellings
            if is_id_like(name) or not counts:
                continue
            avg_len = sum(len(v) for v in counts) / len(counts)
            if avg_len > 60:
                continue  # free text, not labels
            mapping = _spelling_groups(counts)
            if mapping:
                n_rows = sum(counts[v] for v in mapping)
                groups: dict[str, list[str]] = {}
                for variant, canonical in mapping.items():
                    groups.setdefault(canonical, []).append(variant)
                examples = [" / ".join([c, *vs][:4]) + f" → {c}" for c, vs in
                            list(groups.items())[:4]]  # fmt: skip
                issues.append(Issue(
                    f"spelling:{name}", "medium", "Same value, different spellings", name,
                    n_rows,
                    f"{len(groups):,} value{'s are' if len(groups) != 1 else ' is'} written in "
                    f"more than one way in {name}, so they are counted separately.",
                    examples=examples,
                    fix={"type": "merge", "column": name, "mapping": mapping},
                    fix_label=f"Merge into the most common spelling ({n_rows:,} rows)",
                ))  # fmt: skip

        if empty_columns:
            issues.append(Issue(
                "empty-columns", "low", "Empty columns", None, len(empty_columns),
                f"{len(empty_columns):,} column{'s have' if len(empty_columns) != 1 else ' has'}"
                " no values at all.",
                examples=empty_columns[:8],
                fix={"type": "drop_columns", "columns": empty_columns},
                fix_label="Remove the empty columns",
            ))  # fmt: skip
        for name, n in sorted(blanks, key=lambda b: -b[1])[:MAX_BLANK_ISSUES]:
            share = n / total
            issues.append(Issue(
                f"blank:{name}", "medium" if share >= 0.2 and is_id_like(name) else "low",
                "Missing values", name, n,
                f"{n:,} of {total:,} rows ({share:.0%}) have no {name}. They are left out of "
                f"anything grouped or filtered by {name}.",
            ))  # fmt: skip

        for name, kind in kinds.items():
            q = quote_ident(name)
            # 6. Dates in the future (not for due dates, deadlines and plans)
            if kind in ("date", "datetime") and not _PLANNED.search(name):
                future = con.execute(
                    f"SELECT count(*), max({q}) FROM t WHERE CAST({q} AS DATE) > ?",  # noqa: S608
                    [today],
                ).fetchone()
                if future and future[0]:
                    issues.append(Issue(
                        f"future:{name}", "medium", "Dates in the future", name, int(future[0]),
                        f"{int(future[0]):,} rows have a {name} after today "
                        f"(latest: {jsonable(future[1])}). Often a typo in the year.",
                    ))  # fmt: skip
            # 7. Extreme values
            if kind == "number" and not is_id_like(name):
                stats = con.execute(
                    f"SELECT quantile_cont({q}, 0.25), quantile_cont({q}, 0.75) FROM t"  # noqa: S608
                ).fetchone()
                if stats and stats[0] is not None and stats[1] is not None:
                    low, high = float(stats[0]), float(stats[1])
                    spread = high - low
                    if spread > 0:
                        lo, hi = low - 3 * spread, high + 3 * spread
                        rows = con.execute(
                            f"SELECT {q} FROM t WHERE {q} < ? OR {q} > ? "  # noqa: S608
                            f"ORDER BY abs({q}) DESC LIMIT 5",
                            [lo, hi],
                        ).fetchall()
                        if rows:
                            n_out = int(_scalar(con, f"SELECT count(*) FROM t WHERE {q} < ? "  # noqa: S608
                                                f"OR {q} > ?", [lo, hi]) or 0)  # fmt: skip
                            issues.append(Issue(
                                f"outlier:{name}", "low", "Unusual values", name, n_out,
                                f"{n_out:,} values of {name} are far outside the usual range "
                                f"({low:,.4g} to {high:,.4g} for the middle half). Check they "
                                "aren't typos (an extra zero, wrong unit).",
                                examples=[str(jsonable(r[0])) for r in rows],
                            ))  # fmt: skip

    order = {"high": 0, "medium": 1, "low": 2}
    issues.sort(key=lambda i: (order[i.severity], -i.rows))
    return issues


def _spelling_groups(counts: dict[str, int]) -> dict[str, str]:
    """{variant: canonical} for values that are the same thing written differently.
    Exact after normalising (case, spaces, punctuation), or near-identical text without
    digits ("Zarmeen Kahn" ~ "Zarmeen Khan"; never "DMA-11" ~ "DMA-12")."""
    by_norm: dict[str, list[str]] = {}
    for v in counts:
        key = _norm(v)
        if key:
            by_norm.setdefault(key, []).append(v)
    # Near-identical normalised forms (typos), letters only.
    keys = sorted(by_norm, key=lambda k: -sum(counts[v] for v in by_norm[k]))
    merged: dict[str, str] = {}
    for i, a in enumerate(keys):
        if a in merged or any(ch.isdigit() for ch in a) or len(a) < 6:
            continue
        for b in keys[i + 1 :]:
            if b in merged or any(ch.isdigit() for ch in b) or len(b) < 6:
                continue
            if fuzz.ratio(a, b) >= 90:
                merged[b] = a
    for b, a in merged.items():
        by_norm[a] += by_norm.pop(b)

    mapping: dict[str, str] = {}
    for variants in by_norm.values():
        if len(variants) < 2:
            continue
        canonical = max(variants, key=lambda v: (counts[v], v == v.strip(), v))
        canonical = re.sub(r"\s+", " ", canonical).strip()
        for v in variants:
            if v != canonical:
                mapping[v] = canonical
    return mapping


def apply_fixes(source: Path, out: Path, fixes: list[dict[str, Any]], columns: list[str]) -> None:
    """Write a fixed copy of the data to `out` (the original file isn't touched)."""
    drop: set[str] = set()
    distinct = False
    exprs: dict[str, str] = {c: quote_ident(c) for c in columns}
    for fx in fixes:
        kind = fx.get("type")
        if kind == "drop_duplicates":
            distinct = True
        elif kind == "drop_columns":
            drop |= {c for c in fx.get("columns", []) if c in exprs}
        elif kind == "trim" and fx.get("column") in exprs:
            col = fx["column"]
            exprs[col] = f"trim(regexp_replace({exprs[col]}, '\\s+', ' ', 'g'))"
        elif kind == "merge" and fx.get("column") in exprs:
            col, mapping = fx["column"], fx.get("mapping") or {}
            if mapping:
                cases = " ".join(f"WHEN {quote_literal(str(k))} THEN {quote_literal(str(v))}"
                                 for k, v in mapping.items())  # fmt: skip
                exprs[col] = (
                    f"CASE CAST({quote_ident(col)} AS VARCHAR) {cases} ELSE {exprs[col]} END"
                )
    keep = [c for c in columns if c not in drop]
    if not keep:
        raise ValueError("Every column would be removed.")
    select = ", ".join(f"{exprs[c]} AS {quote_ident(c)}" for c in keep)
    with duckdb.connect() as con:
        con.execute(
            f"COPY (SELECT {'DISTINCT ' if distinct else ''}{select} "  # noqa: S608
            f"FROM read_parquet({quote_literal(str(source))})) TO {quote_literal(str(out))} "
            "(FORMAT PARQUET, COMPRESSION ZSTD)"
        )


def summarize(issues: list[Issue], name: str) -> str:
    if not issues:
        return (f"“{name}” looks healthy: no duplicates, spelling variants, stray "
                "spaces, empty columns or unusual values found.")  # fmt: skip
    counts = Counter(i.severity for i in issues)
    fixable = sum(1 for i in issues if i.fix)
    worst = ", ".join(i.title.lower() + (f" in {i.column}" if i.column else "")
                      for i in issues[:3])  # fmt: skip
    text = (f"I found {len(issues)} thing{'s' if len(issues) != 1 else ''} to check in "
            f"“{name}” ({counts['high']} important, {counts['medium']} worth a look, "
            f"{counts['low']} minor). Top: {worst}.")  # fmt: skip
    if fixable:
        text += (f" {fixable} can be fixed automatically: pick them below and press Fix, or "
                 "say “fix them”. The current version is kept, so you can compare.")  # fmt: skip
    return text
