"""Turn an uploaded table file into a clean Parquet file plus a profile.

Supported: delimited text (.csv .tsv .txt), spreadsheets (.xlsx .xls .ods), .json,
.parquet and tables inside .pdf files. Every reader produces rows of cells that go
through the same cleanup (real header row, typed columns, no trailing Total rows).
"""

import csv
import itertools
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

import duckdb
import pandas as pd

from app.connectors.profiling import DatasetProfile, profile_parquet
from app.query.sql import quote_ident, quote_literal

FileFormat = Literal[
    "csv", "tsv", "txt", "xlsx", "xls", "ods", "json", "parquet", "pdf", "png", "jpg", "webp",
    "docx", "md",
]  # fmt: skip
IMAGE_MIME = {"png": "image/png", "jpg": "image/jpeg", "webp": "image/webp"}
MAX_SCANNED_PAGES = 5  # each page is one vision call; keep uploads under proxy time limits

# Copies the table in an image into rows of text cells (an AI vision model, injected by
# the API so this module stays free of LLM code). Raises IngestError when it can't.
Transcriber = Callable[[bytes, str], list[list[str]]]

SUFFIXES: dict[str, FileFormat] = {
    ".csv": "csv",
    ".tsv": "tsv",
    ".tab": "tsv",
    ".txt": "txt",
    ".xlsx": "xlsx",
    ".xlsm": "xlsx",
    ".xls": "xls",
    ".ods": "ods",
    ".json": "json",
    ".jsonl": "json",
    ".ndjson": "json",
    ".parquet": "parquet",
    ".pdf": "pdf",
    ".png": "png",
    ".jpg": "jpg",
    ".jpeg": "jpg",
    ".webp": "webp",
    ".docx": "docx",
    ".md": "md",
    ".markdown": "md",
}
SPREADSHEET_ENGINES = {"xlsx": "openpyxl", "xls": "calamine", "ods": "calamine"}
MAX_PDF_PAGES = 300

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
    extraction: str | None = None  # "ai_vision" when an AI model read the table from an image


def detect_format(filename: str) -> FileFormat | None:
    return SUFFIXES.get(Path(filename).suffix.lower())


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


def _sniff_delimiter(sample: str, default: str) -> str:
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return default


def _ingest_csv(src: Path, out: Path, default_delimiter: str = ",") -> str:
    encoding = _ensure_utf8(src)
    if not src.read_bytes().strip():
        raise IngestError("The file is empty.")
    with src.open(newline="", encoding="utf-8") as fh:
        delimiter = _sniff_delimiter(fh.read(64 * 1024), default_delimiter)
        fh.seek(0)
        reader = csv.reader(fh, delimiter=delimiter)
        first_lines = list(itertools.islice(reader, HEADER_SCAN_ROWS + 6))
    skip = find_header_row([list(r) for r in first_lines])  # title lines above the header
    source = (
        f"read_csv({quote_literal(str(src))}, auto_detect=true, sample_size=-1, header=true, "
        f"skip={int(skip)}, delim={quote_literal(delimiter)})"
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


def frame_from_rows(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw cells (no header yet) -> typed table: real header, no padding, no Total rows."""
    raw = raw.dropna(how="all").reset_index(drop=True)
    if raw.empty:
        raise IngestError("The table is empty.")
    header = find_header_row(raw.head(HEADER_SCAN_ROWS + 6).values.tolist())
    names = ["" if _is_blank(v) else str(v) for v in raw.iloc[header]]
    df = raw.iloc[header + 1 :].reset_index(drop=True)
    df.columns = pd.Index(clean_column_names(names))

    # Drop columns that are blank in the header and empty below (sheet padding).
    padding = [c for c, n in zip(df.columns, names, strict=True) if not n and df[c].isna().all()]
    df = _drop_total_rows(df.drop(columns=padding))
    for col in df.columns:
        df[col] = _typed_column(df[col])
    return df


def _spreadsheet_frame(
    src: Path, fmt: str, sheet: str | None
) -> tuple[pd.DataFrame, str, list[str]]:
    try:
        book = pd.ExcelFile(src, engine=SPREADSHEET_ENGINES[fmt])  # type: ignore[arg-type]
    except Exception as exc:
        raise IngestError(f"Could not open the spreadsheet. Is it a valid .{fmt} file?") from exc
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
    return frame_from_rows(raw), chosen, sheets


def write_frame(df: pd.DataFrame, out: Path) -> None:
    _check_shape(len(df), len(df.columns))
    try:
        with duckdb.connect() as con:
            con.register("frame_src", df)
            _write_parquet(con, "frame_src", out)
    except duckdb.Error as exc:
        raise IngestError(f"Could not convert the table: {_first_line(exc)}") from exc


# --- JSON and Parquet -------------------------------------------------------------------


def _json_records(src: Path) -> list[Any]:
    text = src.read_text(encoding="utf-8-sig", errors="replace").strip()
    if not text:
        raise IngestError("The file is empty.")
    data: Any
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:  # JSON Lines: one object per line
            data = [json.loads(line) for line in text.splitlines() if line.strip()]
        except json.JSONDecodeError as exc:
            raise IngestError(f"This isn't valid JSON (line {exc.lineno}).") from exc
    if isinstance(data, dict):
        # {"data": [...]} style wrappers: use the biggest list of objects inside.
        lists = [v for v in data.values() if isinstance(v, list) and v and isinstance(v[0], dict)]
        data = max(lists, key=len) if lists else [data]
    if not isinstance(data, list) or not data:
        raise IngestError("The JSON doesn't contain a list of records.")
    return data


def _json_frame(src: Path) -> pd.DataFrame:
    records = _json_records(src)
    if all(isinstance(r, list) for r in records):
        return frame_from_rows(pd.DataFrame(records))  # [[header...], [row...], ...]
    if not all(isinstance(r, dict) for r in records):
        raise IngestError("The JSON records must all be objects (or all be lists).")
    df = pd.json_normalize(records, sep=".")  # nested objects -> "address.city"
    df.columns = pd.Index(clean_column_names([str(c) for c in df.columns]))
    for col in df.columns:
        # Lists/objects left in cells are kept as JSON text.
        df[col] = df[col].map(lambda v: json.dumps(v) if isinstance(v, list | dict) else v)
        df[col] = _typed_column(df[col])
    return df


def _ingest_parquet(src: Path, out: Path) -> None:
    try:
        with duckdb.connect() as con:
            _write_parquet(con, f"read_parquet({quote_literal(str(src))})", out)
    except duckdb.Error as exc:
        raise IngestError(f"Could not read the Parquet file: {_first_line(exc)}") from exc


# --- PDF tables -------------------------------------------------------------------------


@dataclass
class PdfTable:
    first_page: int
    last_page: int
    rows: list[list[str]]

    @property
    def name(self) -> str:
        if self.first_page == self.last_page:
            return f"Table (page {self.first_page})"
        return f"Table (pages {self.first_page}-{self.last_page})"


def _clean_cell(v: Any) -> str:
    return re.sub(r"\s+", " ", str(v)).strip() if v is not None else ""


def pdf_tables(src: Path) -> tuple[list[PdfTable], int]:
    """Tables found in the PDF (continuations across pages joined), and its text length."""
    import pdfplumber  # imported lazily: only needed for PDFs

    try:
        pdf = pdfplumber.open(src)
    except Exception as exc:
        raise IngestError("Could not open the PDF. Is it a valid, unencrypted .pdf?") from exc
    tables: list[PdfTable] = []
    text_chars = 0
    with pdf:
        for number, page in enumerate(pdf.pages[:MAX_PDF_PAGES], start=1):
            text_chars += len((page.extract_text() or "").strip())
            for raw in page.extract_tables():
                rows = [[_clean_cell(c) for c in r] for r in raw]
                rows = [r for r in rows if any(r)]
                if len(rows) < 2 or max(len(r) for r in rows) < 2:
                    continue
                prev = tables[-1] if tables else None
                if (
                    prev is not None
                    and prev.last_page == number - 1
                    and len(rows[0]) == len(prev.rows[0])
                ):
                    # A table continuing from the previous page (repeated header dropped).
                    prev.rows.extend(rows[1:] if rows[0] == prev.rows[0] else rows)
                    prev.last_page = number
                else:
                    tables.append(PdfTable(number, number, rows))
    return tables, text_chars


def _rows_frame(rows: list[list[str]]) -> pd.DataFrame:
    rows = [r for r in rows if any(c.strip() for c in r)]
    if len(rows) < 2:
        raise IngestError("No table found in the image.")
    width = max(len(r) for r in rows)
    raw = pd.DataFrame([[c or None for c in r] + [None] * (width - len(r)) for r in rows])
    return frame_from_rows(raw)


def _image_frame(image: bytes, mime: str, transcribe: Transcriber | None) -> pd.DataFrame:
    if transcribe is None:
        raise IngestError(
            "Reading tables from images needs a vision model. Ask your admin to set "
            "GEMINI_API_KEY (or VISION_PROVIDER)."
        )
    return _rows_frame(transcribe(image, mime))


def _scanned_pdf_frame(src: Path, transcribe: Transcriber) -> pd.DataFrame:
    """Render the first pages to images and let the vision model read the table."""
    import io

    import pypdfium2

    rows: list[list[str]] = []
    pdf = pypdfium2.PdfDocument(src)
    try:
        for index in range(min(len(pdf), MAX_SCANNED_PAGES)):
            buf = io.BytesIO()
            pdf[index].render(scale=2).to_pil().save(buf, format="PNG")
            page_rows = [r for r in transcribe(buf.getvalue(), "image/png") if any(r)]
            if rows and page_rows and page_rows[0] == rows[0]:
                page_rows = page_rows[1:]  # header repeated on the next page
            rows.extend(page_rows)
    finally:
        pdf.close()
    return _rows_frame(rows)


def pdf_is_mostly_tables(src: Path) -> bool:
    """True when the PDF is mainly tables (or scanned, so tables come from vision)."""
    tables, text_chars = pdf_tables(src)
    if text_chars == 0:
        return True
    table_chars = sum(len(c) for t in tables for r in t.rows for c in r)
    return bool(tables) and table_chars >= 0.4 * text_chars


def _pdf_frame(
    src: Path, sheet: str | None, transcribe: Transcriber | None = None
) -> tuple[pd.DataFrame, str, list[str], str | None]:
    tables, text_chars = pdf_tables(src)
    if not tables:
        if text_chars == 0:
            if transcribe is not None:
                return _scanned_pdf_frame(src, transcribe), "Scanned pages", [], "ai_vision"
            raise IngestError(
                "This PDF has no selectable text (it looks scanned), and no vision model is set "
                "up to read it."
            )
        raise IngestError("No tables found in this PDF.")
    names = [t.name for t in tables]
    if len(set(names)) < len(names):  # several tables on one page
        names = [f"{n} #{i}" for i, n in enumerate(names, start=1)]
    if sheet is not None:
        if sheet not in names:
            raise IngestError(f"Table {sheet!r} not found. Available: {', '.join(names)}")
        chosen = names.index(sheet)
    else:  # the biggest table by default
        chosen = max(range(len(tables)), key=lambda i: len(tables[i].rows) * len(tables[i].rows[0]))
    return _rows_frame(tables[chosen].rows), names[chosen], names, None


def _first_line(exc: Exception) -> str:
    return str(exc).strip().splitlines()[0][:300] if str(exc).strip() else type(exc).__name__


def ingest_file(
    src: Path,
    fmt: FileFormat,
    out: Path,
    sheet: str | None = None,
    transcribe: Transcriber | None = None,
) -> IngestResult:
    """Convert `src` to Parquet at `out` and profile it. Raises IngestError for bad input."""
    extraction: str | None = None
    encoding: str | None = None
    sheet_name: str | None = None
    sheet_names: list[str] = []
    if fmt in ("csv", "tsv", "txt"):
        encoding = _ingest_csv(src, out, default_delimiter="\t" if fmt == "tsv" else ",")
    elif fmt in SPREADSHEET_ENGINES:
        df, sheet_name, sheet_names = _spreadsheet_frame(src, fmt, sheet)
        write_frame(df, out)
    elif fmt == "pdf":
        df, sheet_name, sheet_names, extraction = _pdf_frame(src, sheet, transcribe)
        write_frame(df, out)
    elif fmt in IMAGE_MIME:
        write_frame(_image_frame(src.read_bytes(), IMAGE_MIME[fmt], transcribe), out)
        extraction = "ai_vision"
    elif fmt == "json":
        write_frame(_json_frame(src), out)
    elif fmt == "parquet":
        _ingest_parquet(src, out)
    else:
        raise IngestError(f"Unsupported file type: .{fmt}")

    profile = profile_parquet(out)
    _check_shape(profile.row_count, len(profile.columns))
    return IngestResult(profile, sheet_name, sheet_names, encoding, extraction)
