"""Turn an uploaded CSV/XLSX file into a clean Parquet file plus a profile."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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
    source = f"read_csv({quote_literal(str(src))}, auto_detect=true, sample_size=-1, header=true)"
    try:
        with duckdb.connect() as con:
            _write_parquet(con, source, out)
    except duckdb.Error as exc:
        raise IngestError(f"Could not parse the CSV file: {_first_line(exc)}") from exc
    return encoding


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
        df = pd.read_excel(book, sheet_name=chosen, nrows=MAX_ROWS + 1)

    df = df.dropna(how="all")
    blank_cols = [c for c in df.columns if str(c).startswith("Unnamed:") and df[c].isna().all()]
    df = df.drop(columns=blank_cols)
    df.columns = pd.Index([str(c) for c in df.columns])

    for col in df.columns:
        if df[col].dtype == object:
            inferred = pd.api.types.infer_dtype(df[col], skipna=True)
            if inferred not in ("string", "empty", "date", "datetime", "boolean"):
                # Mixed cells (e.g. numbers and text): keep everything, as text.
                df[col] = df[col].map(lambda v: None if pd.isna(v) else str(v))
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
