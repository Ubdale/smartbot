from pathlib import Path

import duckdb
import pytest

from app.connectors.csv_xlsx import clean_column_names, detect_format
from app.connectors.profiling import jsonable, kind_of
from app.query.sql import quote_ident, quote_literal
from app.storage import DatasetStorage

WS = "11111111-1111-4111-8111-111111111111"
DS = "22222222-2222-4222-8222-222222222222"


def test_clean_column_names() -> None:
    assert clean_column_names([" a ", "", "A", "Unnamed: 3", "x  y", "a"]) == [
        "a",
        "column_2",
        "A_2",
        "column_4",
        "x y",
        "a_3",
    ]


def test_detect_format() -> None:
    assert detect_format("Report.CSV") == "csv"
    assert detect_format("book.xlsx") == "xlsx"
    assert detect_format("book.xls") == "xls"
    assert detect_format("Data.JSONL") == "json"
    assert detect_format("report.pdf") == "pdf"
    assert detect_format("notes.docx") == "docx"
    assert detect_format("deck.pptx") is None
    assert detect_format("noext") is None


@pytest.mark.parametrize(
    ("t", "kind"),
    [
        ("BIGINT", "number"),
        ("DECIMAL(18,2)", "number"),
        ("DOUBLE", "number"),
        ("UBIGINT", "number"),
        ("DATE", "date"),
        ("TIMESTAMP WITH TIME ZONE", "datetime"),
        ("TIMESTAMP_NS", "datetime"),
        ("BOOLEAN", "boolean"),
        ("VARCHAR", "text"),
        ("TIME", "text"),
    ],
)
def test_kind_of(t: str, kind: str) -> None:
    assert kind_of(t) == kind


def test_jsonable() -> None:
    assert jsonable(float("nan")) is None
    assert jsonable(float("inf")) is None
    assert jsonable(1.5) == 1.5


def test_quote_ident_survives_hostile_names() -> None:
    hostile = 'x"; DROP TABLE t; --'
    with duckdb.connect() as con:
        con.execute(f"CREATE TABLE t ({quote_ident(hostile)} INTEGER)")
        con.execute("INSERT INTO t VALUES (1)")
        row = con.execute(f"SELECT {quote_ident(hostile)} FROM t").fetchone()
        assert row == (1,)
        assert con.execute("SELECT count(*) FROM t").fetchone() == (1,)
    with pytest.raises(ValueError, match="NUL"):
        quote_ident("a\x00b")
    assert quote_literal("it's") == "'it''s'"


def test_storage_rejects_non_uuid_ids(tmp_path: Path) -> None:
    storage = DatasetStorage(tmp_path)
    assert storage.dataset_dir(WS, DS).is_relative_to(tmp_path.resolve())
    for bad in ("../../etc", "..", WS + "/../x", "not-a-uuid"):
        with pytest.raises(ValueError, match="server-generated"):
            storage.dataset_dir(bad, DS)
        with pytest.raises(ValueError, match="server-generated"):
            storage.dataset_dir(WS, bad)
