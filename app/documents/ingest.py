"""Turn a document (.pdf text, .docx, .txt, .md) into searchable passages.

Passages are stored as a small Parquet file (chunk_id, page, text) next to the
original, so documents reuse the same storage, caching and workspace isolation
as tables.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

from app.connectors.csv_xlsx import MAX_PDF_PAGES, IngestError
from app.query.sql import quote_literal

DOCUMENT_FORMATS = {"pdf", "docx", "txt", "md"}
CHUNK_CHARS = 900
CHUNK_OVERLAP = 150
MAX_CHUNKS = 5000


@dataclass(frozen=True)
class Passage:
    chunk_id: int
    page: int | None  # PDF page number; None for formats without pages
    text: str


@dataclass(frozen=True)
class DocumentResult:
    passages: int
    pages: int | None
    characters: int


def _decode(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise IngestError("Could not read the file's text encoding.")


def _pdf_pages(src: Path) -> list[tuple[int | None, str]]:
    import pdfplumber

    try:
        pdf = pdfplumber.open(src)
    except Exception as exc:
        raise IngestError("Could not open the PDF. Is it a valid, unencrypted .pdf?") from exc
    with pdf:
        return [
            (number, page.extract_text() or "")
            for number, page in enumerate(pdf.pages[:MAX_PDF_PAGES], start=1)
        ]


def _docx_text(src: Path) -> str:
    import docx

    try:
        document = docx.Document(str(src))
    except Exception as exc:
        raise IngestError("Could not open the Word file. Is it a valid .docx?") from exc
    parts = [p.text for p in document.paragraphs]
    for table in document.tables:  # keep table contents searchable, one row per line
        for row in table.rows:
            parts.append(" | ".join(cell.text.strip() for cell in row.cells))
    return "\n\n".join(parts)


def extract_pages(src: Path, fmt: str) -> list[tuple[int | None, str]]:
    if fmt == "pdf":
        return _pdf_pages(src)
    if fmt == "docx":
        return [(None, _docx_text(src))]
    if fmt in ("txt", "md"):
        return [(None, _decode(src.read_bytes()))]
    raise IngestError(f".{fmt} files can't be read as documents.")


def _split(text: str) -> list[str]:
    """~CHUNK_CHARS pieces on paragraph/sentence boundaries, overlapping a little."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces: list[str] = []
    current = ""
    for paragraph in paragraphs:
        units = (
            [paragraph] if len(paragraph) <= CHUNK_CHARS else re.split(r"(?<=[.!?])\s+", paragraph)
        )
        for unit in units:
            while len(unit) > CHUNK_CHARS:  # one enormous sentence: hard split
                pieces.append(unit[:CHUNK_CHARS])
                unit = unit[CHUNK_CHARS - CHUNK_OVERLAP :]
            if current and len(current) + len(unit) + 1 > CHUNK_CHARS:
                pieces.append(current)
                current = current[-CHUNK_OVERLAP:] + " " + unit
            else:
                current = f"{current}\n{unit}" if current else unit
    if current.strip():
        pieces.append(current)
    return [re.sub(r"[ \t]+", " ", p).strip() for p in pieces]


def chunk(pages: list[tuple[int | None, str]]) -> list[Passage]:
    passages: list[Passage] = []
    for page, text in pages:
        for piece in _split(text):
            passages.append(Passage(len(passages) + 1, page, piece))
            if len(passages) >= MAX_CHUNKS:
                return passages
    return passages


def ingest_document(src: Path, fmt: str, out: Path) -> DocumentResult:
    pages = extract_pages(src, fmt)
    passages = chunk(pages)
    characters = sum(len(p.text) for p in passages)
    if not passages or characters < 20:
        if fmt == "pdf":
            raise IngestError(
                "This PDF has no selectable text (it looks scanned). Upload the pages as images."
            )
        raise IngestError("The document has no readable text.")
    frame = pd.DataFrame(
        {
            "chunk_id": [p.chunk_id for p in passages],
            "page": pd.array([p.page for p in passages], dtype="Int64"),
            "text": [p.text for p in passages],
        }
    )
    with duckdb.connect() as con:
        con.register("passages", frame)
        con.execute(f"COPY passages TO {quote_literal(str(out))} (FORMAT parquet)")
    page_numbers = [p for p, _ in pages if p is not None]
    return DocumentResult(len(passages), max(page_numbers) if page_numbers else None, characters)


def load_passages(path: Path) -> list[Passage]:
    with duckdb.connect() as con:
        rows = con.execute(
            "SELECT chunk_id, page, text FROM read_parquet(?) ORDER BY chunk_id", [str(path)]
        ).fetchall()
    return [Passage(int(r[0]), int(r[1]) if r[1] is not None else None, str(r[2])) for r in rows]
