"""Every supported table format ends up as the same clean, typed dataset."""

import json
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pytest
from fpdf import FPDF

from app.connectors.csv_xlsx import IngestError, detect_format, ingest_file

ROWS: list[list[Any]] = [["region", "revenue"], ["North", 120.5], ["South", 80], ["East", 42]]


def ingest(path: Path, sheet: str | None = None) -> tuple[dict[str, Any], int, Any]:
    fmt = detect_format(path.name)
    assert fmt is not None
    res = ingest_file(path, fmt, path.with_name(path.name + ".parquet"), sheet=sheet)
    return {c.name: c for c in res.profile.columns}, res.profile.row_count, res


def assert_regions(cols: dict[str, Any], rows: int) -> None:
    assert list(cols) == ["region", "revenue"]
    assert rows == 3
    assert cols["revenue"].kind == "number"
    assert (cols["revenue"].min, cols["revenue"].max) == (42, 120.5)


# --- delimited text -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "text"),
    [
        ("data.tsv", "region\trevenue\nNorth\t120.5\nSouth\t80\nEast\t42\n"),
        ("data.txt", "region|revenue\nNorth|120.5\nSouth|80\nEast|42\n"),
        ("semicolon.csv", "region;revenue\nNorth;120.5\nSouth;80\nEast;42\n"),  # EU Excel
    ],
)
def test_delimited_variants(tmp_path: Path, name: str, text: str) -> None:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    assert_regions(*ingest(path)[:2])


# --- spreadsheets ---------------------------------------------------------------------------


def test_legacy_xls(tmp_path: Path) -> None:
    import xlwt

    book = xlwt.Workbook()
    sheet = book.add_sheet("Sales")
    for r, row in enumerate(ROWS):
        for c, v in enumerate(row):
            sheet.write(r, c, v)
    path = tmp_path / "old.xls"
    book.save(str(path))
    cols, rows, res = ingest(path)
    assert_regions(cols, rows)
    assert res.sheet_name == "Sales"


def test_open_document_spreadsheet(tmp_path: Path) -> None:
    path = tmp_path / "data.ods"
    pd.DataFrame(ROWS[1:], columns=ROWS[0]).to_excel(path, engine="odf", index=False)  # type: ignore[arg-type]
    assert_regions(*ingest(path)[:2])


# --- JSON / Parquet -------------------------------------------------------------------------


RECORDS = [{"region": r, "revenue": v} for r, v in ROWS[1:]]


@pytest.mark.parametrize(
    "content",
    [
        json.dumps(RECORDS),
        json.dumps({"meta": {"source": "crm"}, "data": RECORDS}),  # wrapped list
        "\n".join(json.dumps(r) for r in RECORDS),  # JSON Lines
        json.dumps(ROWS),  # list of lists with a header row
    ],
)
def test_json_shapes(tmp_path: Path, content: str) -> None:
    path = tmp_path / "data.json"
    path.write_text(content, encoding="utf-8")
    assert_regions(*ingest(path)[:2])


def test_nested_json_is_flattened(tmp_path: Path) -> None:
    path = tmp_path / "nested.json"
    path.write_text(
        json.dumps(
            [
                {"id": 1, "customer": {"name": "Apex", "city": "Karachi"}, "tags": ["a", "b"]},
                {"id": 2, "customer": {"name": "Blue", "city": "Lahore"}, "tags": []},
            ]
        ),
        encoding="utf-8",
    )
    cols, rows, _ = ingest(path)
    assert set(cols) == {"id", "customer.name", "customer.city", "tags"}
    assert cols["tags"].kind == "text"  # lists kept as JSON text
    assert rows == 2


@pytest.mark.parametrize("content", ["", "{not json", "42", '[1, {"a": 2}]'])
def test_bad_json(tmp_path: Path, content: str) -> None:
    path = tmp_path / "bad.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(IngestError):
        ingest(path)


def test_parquet(tmp_path: Path) -> None:
    path = tmp_path / "data.parquet"
    with duckdb.connect() as con:
        con.execute(
            f"COPY (SELECT * FROM (VALUES ('North', 120.5), ('South', 80.0), ('East', 42.0)) "
            f"t(region, revenue)) TO '{path.as_posix()}' (FORMAT parquet)"
        )
    assert_regions(*ingest(path)[:2])


# --- PDF tables ------------------------------------------------------------------------------


def pdf_with_tables(path: Path, tables: list[list[list[str]]], *, page_per_table: bool) -> Path:
    pdf = FPDF()
    pdf.set_font("Helvetica", size=10)
    pdf.add_page()
    pdf.cell(text="Quarterly report - Acme Bank Limited", new_x="LMARGIN", new_y="NEXT")
    for i, rows in enumerate(tables):
        if i and page_per_table:
            pdf.add_page()
        pdf.ln(4)
        with pdf.table() as table:
            for row in rows:
                r = table.row()
                for cell in row:
                    r.cell(cell)
    pdf.output(str(path))
    return path


def test_pdf_table(tmp_path: Path) -> None:
    rows = [[str(c) for c in r] for r in ROWS]
    cols, n, res = ingest(pdf_with_tables(tmp_path / "r.pdf", [rows], page_per_table=False))
    assert_regions(cols, n)
    assert res.sheet_name == "Table (page 1)"


def test_pdf_table_continuing_on_next_page_is_joined(tmp_path: Path) -> None:
    header = ["region", "revenue"]
    part1 = [header, ["North", "120.5"], ["South", "80"]]
    part2 = [header, ["East", "42"]]  # header repeated on the next page
    cols, n, res = ingest(pdf_with_tables(tmp_path / "r.pdf", [part1, part2], page_per_table=True))
    assert_regions(cols, n)
    assert res.sheet_name == "Table (pages 1-2)"


def test_pdf_choose_among_tables(tmp_path: Path) -> None:
    small = [["team", "size"], ["A", "3"], ["B", "4"]]
    big = [[str(c) for c in r] for r in ROWS] + [["West", "7"]]
    path = pdf_with_tables(tmp_path / "r.pdf", [small, [["x"]], big[:1] + big[1:]],
                           page_per_table=True)  # fmt: skip
    _, n, res = ingest(path)  # biggest table by default
    assert n == 4
    assert len(res.sheet_names) == 2
    _, n_small, _ = ingest(path, sheet=res.sheet_names[0])
    assert n_small == 2


def test_pdf_without_tables(tmp_path: Path) -> None:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(0, 8, "Just a letter. No tables here at all.")
    path = tmp_path / "letter.pdf"
    pdf.output(str(path))
    with pytest.raises(IngestError, match="No tables"):
        ingest(path)


def test_scanned_pdf_has_a_clear_message(tmp_path: Path) -> None:
    pdf = FPDF()
    pdf.add_page()  # a page with no text layer, like a scan
    path = tmp_path / "scan.pdf"
    pdf.output(str(path))
    with pytest.raises(IngestError, match="scanned"):
        ingest(path)


def test_not_a_pdf(tmp_path: Path) -> None:
    path = tmp_path / "fake.pdf"
    path.write_bytes(b"hello")
    with pytest.raises(IngestError, match="Could not open the PDF"):
        ingest(path)
