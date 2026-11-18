"""Turn an uploaded CSV/XLSX file into a clean Parquet file plus a profile."""

import csv
import itertools
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

import duckdb
import pandas as pd

from app.connectors.profiling import DatasetProfile, profile_parquet
from app.query.sql import quote_ident, quote_literal

FileFormat = Literal["csv", "xlsx"]

MAX_COLUMNS = 500
MAX_ROWS = 2_000_000
MAX_COLUMN_NAME = 120


class IngestError(ValueError):
    """The file can't be turned into a dataset; the message is safe to show users."""


@dataclass(frozen=True)
class IngestResult:
    profile: DatasetProfile
    sheet_name: str | None
    sheet_names: list[str]
    encoding: str | None


def detect_format(filename: str) -> FileFormat | None:
    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix == ".xlsx":
        return "xlsx"
    return None


def clean_column_names(names: list[str]) -> list[str]:
    """Trim, collapse whitespace, fill blanks, cap length and de-duplicate (case-insensitive)."""
    cleaned: list[str] = []
    seen: set[str] = set()
    for i, raw in enumerate(names, start=1):
        name = re.sub(r"\s+", " ", str(raw)).strip()[:MAX_COLUMN_NAME]
        if not name or name.lower() == "nan" or name.startswith("Unnamed:"):
            name = f"column_{i}"
        candidate, n = name, 2
        while candidate.lower() in seen:
            candidate, n = f"{name}_{n}", n + 1
        seen.add(candidate.lower())
        cleaned.append(candidate)
    return cleaned


def _ensure_utf8(path: Path) -> str:
    """Make the CSV UTF-8 in place. Returns the encoding it was in."""
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        path.write_bytes(raw[3:])
        return "utf-8-sig"
    try:
        raw.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        pass
    # Excel on Windows saves "CSV" as cp1252; latin-1 never fails as a last resort.
    for enc in ("cp1252", "latin-1"):
        try:
            text = raw.decode(enc)
        except UnicodeDecodeError:
            continue
        path.write_text(text, encoding="utf-8", newline="")
        return enc
    raise IngestError("Could not read the file's text encoding. Please save it as UTF-8 CSV.")


def _check_shape(n_rows: int, n_cols: int) -> None:
    if n_cols == 0:
        raise IngestError("The file has no columns.")
    if n_cols > MAX_COLUMNS:
        raise IngestError(f"The file has {n_cols} columns; the limit is {MAX_COLUMNS}.")
    if n_rows == 0:
        raise IngestError("The file has a header row but no data rows.")
    if n_rows > MAX_ROWS:
        raise IngestError(f"The file has more than {MAX_ROWS:,} rows.")


def _write_parquet(con: duckdb.DuckDBPyConnection, source_sql: str, out: Path) -> None:
    """Copy `source_sql` (a relation with original names) to Parquet with cleaned names."""
    described = con.execute(f"DESCRIBE SELECT * FROM {source_sql}").fetchall()  # noqa: S608
    original = [str(r[0]) for r in described]
    _check_shape(1, len(original))  # column checks before scanning rows
    cleaned = clean_column_names(original)
    select = ", ".join(
        f"{quote_ident(o)} AS {quote_ident(c)}" for o, c in zip(original, cleaned, strict=True)
    )
    con.execute(
        f"COPY (SELECT {select} FROM {source_sql}) TO {quote_literal(str(out))} (FORMAT parquet)"  # noqa: S608
    )


def _ingest_csv(src: Path, out: Path) -> str:
    encoding = _ensure_utf8(src)
    if not src.read_bytes().strip():
        raise IngestError("The file is empty.")
    with src.open(newline="", encoding="utf-8") as fh:
        first_lines = list(itertools.islice(csv.reader(fh), HEADER_SCAN_ROWS + 6))
    skip = find_header_row([list(r) for r in first_lines])  # title lines above the header
    source = (
        f"read_csv({quote_literal(str(src))}, auto_detect=true, sample_size=-1, header=true, "
        f"skip={int(skip)})"
    )
    try:
        with duckdb.connect() as con:
            _write_parquet(con, source, out)
    except duckdb.Error as exc:
        raise IngestError(f"Could not parse the CSV file: {_first_line(exc)}") from exc
    return encoding


# --- report-style files: title lines above the real header -----------------------------

HEADER_SCAN_ROWS = 30
_NUMERIC_TEXT = re.compile(r"^-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?$")
_TOTAL_ROW = re.compile(r"^\s*(grand\s+|sub\s*)?totals?\b", re.I)


def _is_blank(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str):
        return not v.strip()
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _looks_numeric(v: Any) -> bool:
    if isinstance(v, bool):
        return False
    if isinstance(v, int | float):
        return True
    return isinstance(v, str) and bool(_NUMERIC_TEXT.match(v.strip()))


def _is_label(v: Any) -> bool:
    return isinstance(v, str) and bool(v.strip()) and not _looks_numeric(v)


def find_header_row(rows: list[list[Any]]) -> int:
    """Index of the real header row in a sheet that may start with title lines.

    A header is a row of >= 2 text labels whose columns the rows below keep filling.
    Title rows ("Company Ltd", "Headcount by Location", "Date: ...") are sparser and
    don't line up with the data. Returns 0 (first row) when unsure, as before.
    """
    filled = [{j for j, v in enumerate(r) if not _is_blank(v)} for r in rows]

    def coverage(i: int) -> tuple[float, int]:
        below = [f for f in filled[i + 1 :] if f][:5]
        if not below or not filled[i]:
            return 0.0, 0
        cov = sum(len(filled[i] & b) / len(filled[i]) for b in below) / len(below)
        return cov, max(len(b) for b in below)

    for i in range(min(len(rows), HEADER_SCAN_ROWS)):
        cols = filled[i]
        if len(cols) < 2:
            continue
        cov, widest = coverage(i)
        if cov >= 0.8 and any(
            len(filled[k]) >= len(cols) and coverage(k)[0] >= 0.8 for k in range(i)
        ):
            return 0  # an earlier, equally wide row already lines up with the data
        if all(_is_label(rows[i][j]) for j in cols) and cov >= 0.8 and len(cols) >= widest / 2:
            return i
    return 0


def _leading_zero(v: Any) -> bool:
    """'00123' is an ID, not a number: keep it as text."""
    text = str(v).strip().lstrip("-")
    return isinstance(v, str) and len(text) > 1 and text[0] == "0" and text[1] != "."


def _typed_column(s: pd.Series) -> pd.Series:
    """Give a column of raw cells one type: number, date/time or text."""
    values = [v for v in s if not _is_blank(v)]
    cleaned = s.map(lambda v: None if _is_blank(v) else v)
    if not values:
        return cleaned
    if all(isinstance(v, bool) for v in values):
        return cleaned
    if all(_looks_numeric(v) and not _leading_zero(v) for v in values):
        # Numbers, including "2,579" stored as text in the sheet.
        as_text = cleaned.map(lambda v: v.replace(",", "") if isinstance(v, str) else v)
        return pd.to_numeric(as_text, errors="coerce")
    if all(isinstance(v, datetime | date) for v in values):
        return pd.to_datetime(cleaned, errors="coerce")
    # Text, or mixed cells (numbers and text): keep everything, as text.
    return cleaned.map(lambda v: None if v is None else str(v).strip())


def _drop_total_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Report sheets often end with a 'Total' row, which would be counted twice."""
    while len(df):
        first = next((v for v in df.iloc[-1] if not _is_blank(v)), None)
        if not (isinstance(first, str) and _TOTAL_ROW.match(first)):
            break
        df = df.iloc[:-1]
    return df


def _xlsx_frame(src: Path, sheet: str | None) -> tuple[pd.DataFrame, str, list[str]]:
    try:
        book = pd.ExcelFile(src, engine="openpyxl")
    except Exception as exc:
        raise IngestError("Could not open the Excel file. Is it a valid .xlsx?") from exc
    with book:
        sheets = [str(s) for s in book.sheet_names]
        if not sheets:
            raise IngestError("The workbook has no sheets.")
        chosen = sheet if sheet is not None else sheets[0]
        if chosen not in sheets:
            raise IngestError(f"Sheet {chosen!r} not found. Available: {', '.join(sheets)}")
        raw = pd.read_excel(
            book, sheet_name=chosen, header=None, nrows=MAX_ROWS + HEADER_SCAN_ROWS + 1
        )

    raw = raw.dropna(how="all").reset_index(drop=True)
    if raw.empty:
        raise IngestError("The sheet is empty.")
    header = find_header_row(raw.head(HEADER_SCAN_ROWS + 6).values.tolist())
    names = ["" if _is_blank(v) else str(v) for v in raw.iloc[header]]
    df = raw.iloc[header + 1 :].reset_index(drop=True)
    df.columns = pd.Index(clean_column_names(names))

    # Drop columns that are blank in the header and empty below (sheet padding).
    padding = [c for c, n in zip(df.columns, names, strict=True) if not n and df[c].isna().all()]
    df = _drop_total_rows(df.drop(columns=padding))
    for col in df.columns:
        df[col] = _typed_column(df[col])
    return df, chosen, sheets


def _ingest_xlsx(src: Path, out: Path, sheet: str | None) -> tuple[str, list[str]]:
    df, chosen, sheets = _xlsx_frame(src, sheet)
    _check_shape(len(df), len(df.columns))
    try:
        with duckdb.connect() as con:
            con.register("xlsx_src", df)
            _write_parquet(con, "xlsx_src", out)
    except duckdb.Error as exc:
        raise IngestError(f"Could not convert the sheet: {_first_line(exc)}") from exc
    return chosen, sheets


def _first_line(exc: Exception) -> str:
    return str(exc).strip().splitlines()[0][:300] if str(exc).strip() else type(exc).__name__


def ingest_file(src: Path, fmt: FileFormat, out: Path, sheet: str | None = None) -> IngestResult:
    """Convert `src` to Parquet at `out` and profile it. Raises IngestError for bad input."""
    if fmt == "csv":
        encoding: str | None = _ingest_csv(src, out)
        sheet_name: str | None = None
        sheet_names: list[str] = []
    else:
        encoding = None
        sheet_name, sheet_names = _ingest_xlsx(src, out, sheet)

    profile = profile_parquet(out)
    _check_shape(profile.row_count, len(profile.columns))
    return IngestResult(profile, sheet_name, sheet_names, encoding)
