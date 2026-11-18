"""Report-style spreadsheets: title lines above the header, totals, numbers stored as text."""

from pathlib import Path
from typing import Any

import pytest
from openpyxl import Workbook

from app.connectors.csv_xlsx import find_header_row, ingest_file


def xlsx(path: Path, rows: list[list[Any]]) -> Path:
    wb = Workbook()
    ws = wb.active
    assert ws is not None
    for r in rows:
        ws.append(r)
    wb.save(path)
    return path


def profile(path: Path, fmt: str = "xlsx") -> tuple[dict[str, Any], int]:
    res = ingest_file(path, fmt, path.with_suffix(".parquet"))  # type: ignore[arg-type]
    return {c.name: c for c in res.profile.columns}, res.profile.row_count


# Shaped like a real "Headcount By Location" export.
REPORT: list[list[Any]] = [
    ["Acme Bank Limited", None, "Date: 11/8/2026"],
    ["Headcount By Location", None, None],
    [None, None, None],
    ["Categories", "Count", None],
    ["Karachi", "2579", None],
    ["Lahore", "851", None],
    ["Islamabad", "1,204", None],
    ["Total", "4,634", None],
]


def test_report_xlsx_finds_header_drops_total_and_types_numbers(tmp_path: Path) -> None:
    cols, rows = profile(xlsx(tmp_path / "r.xlsx", REPORT))
    assert list(cols) == ["Categories", "Count"]
    assert rows == 3  # the Total row is gone, so sums aren't doubled
    assert cols["Count"].kind == "number"
    assert (cols["Count"].min, cols["Count"].max) == (851, 2579)
    assert cols["Categories"].kind == "text"


def test_report_csv_skips_title_lines(tmp_path: Path) -> None:
    path = tmp_path / "r.csv"
    path.write_text(
        "Acme Bank Limited,,Date: 11/8/2026\nHeadcount By Location,,\n"
        "Categories,Count\nKarachi,2579\nLahore,851\n",
        encoding="utf-8",
    )
    cols, rows = profile(path, "csv")
    assert list(cols) == ["Categories", "Count"]
    assert rows == 2
    assert cols["Count"].kind == "number"


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        (REPORT, 3),
        ([["name", "amount"], ["a", 1], ["b", 2]], 0),  # normal table
        ([["Title only"], ["name", "amount"], ["a", 1]], 1),
        # Numeric header names (years): the first row is still the header.
        ([["region", 2024, 2025], ["North", "x", "y"], ["South", "z", "w"]], 0),
        ([["a"], ["b"], ["c"]], 0),  # single column: unchanged behaviour
    ],
)
def test_find_header_row(rows: list[list[Any]], expected: int) -> None:
    assert find_header_row(rows) == expected


def test_ids_with_leading_zeros_stay_text(tmp_path: Path) -> None:
    cols, _ = profile(xlsx(tmp_path / "ids.xlsx", [["id", "n"], ["00123", 1], ["00456", 2]]))
    assert cols["id"].kind == "text"
    assert cols["id"].samples == ["00123", "00456"]


def test_normal_sheet_is_unchanged(tmp_path: Path) -> None:
    cols, rows = profile(
        xlsx(tmp_path / "n.xlsx", [["Team", "Revenue"], ["North", 10.5], ["South", 20.25]])
    )
    assert list(cols) == ["Team", "Revenue"]
    assert rows == 2
    assert cols["Revenue"].kind == "number"


def test_only_trailing_total_rows_are_dropped(tmp_path: Path) -> None:
    rows: list[list[Any]] = [
        ["item", "qty"],
        ["Total Care Kit", 3],
        ["Soap", 2],
        ["Grand Total", 5],
    ]
    cols, n = profile(xlsx(tmp_path / "t.xlsx", rows))
    assert n == 2  # "Total Care Kit" (first row) is data and stays
    assert cols["item"].samples[0] == "Total Care Kit"
