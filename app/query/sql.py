"""Tiny helpers for building DuckDB SQL safely.

User- and LLM-supplied *values* must always go through DuckDB parameters (`?`).
These helpers exist only for identifiers (column names) and for server-generated
file paths in statements that don't accept parameters (e.g. COPY ... TO).
"""


def quote_ident(name: str) -> str:
    """Quote a column/table identifier: wrap in double quotes, double any inside."""
    if "\x00" in name:
        raise ValueError("identifier contains NUL byte")
    return '"' + name.replace('"', '""') + '"'


def quote_literal(value: str) -> str:
    """Quote a string literal. Only for server-generated values such as file paths."""
    if "\x00" in value:
        raise ValueError("literal contains NUL byte")
    return "'" + value.replace("'", "''") + "'"
