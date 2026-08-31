"""SQL limit detection and insertion."""

from collections.abc import Callable
from typing import cast

import sqlglot
from sqlglot import exp


def has_limit_clause(stmt: exp.Expression) -> bool:
    """Check if a statement already includes a LIMIT/TOP/FETCH clause."""
    limit_types: list[type[exp.Expression]] = [exp.Limit, exp.Fetch]
    top_type = getattr(exp, "Top", None)
    if isinstance(top_type, type):
        limit_types.append(top_type)
    return any(isinstance(node, tuple(limit_types)) for node in stmt.walk())


def add_limit(sql: str, dialect: str = "ansi", limit: int = 100) -> str:
    """Add LIMIT clause to query if not already present.

    Args:
        sql: SQL query
        dialect: SQL dialect for proper rendering
        limit: Maximum number of rows to return

    Returns:
        SQL with LIMIT clause added (or original if LIMIT already exists)
    """
    # Strip trailing semicolon to ensure clean parsing and modification
    # This handles cases where models generate SQL with a trailing semicolon
    sql = sql.strip().rstrip(";")

    try:
        statements = sqlglot.parse(sql, read=dialect)
        if len(statements) != 1:
            return sql

        raw_stmt = statements[0]
        if raw_stmt is None:
            return sql
        stmt = cast(exp.Expression, raw_stmt)

        # Check if LIMIT/TOP/FETCH already exists
        if has_limit_clause(stmt):
            return stmt.sql(dialect=dialect)

        # Add LIMIT - sqlglot will render appropriately for dialect
        # (LIMIT for most, TOP for SQL Server, FETCH FIRST for Oracle)
        limit_method: Callable[[int], exp.Expression] | None = getattr(
            stmt, "limit", None
        )
        if limit_method is not None:
            limited_stmt = limit_method(limit)
            return limited_stmt.sql(dialect=dialect)
        return stmt.sql(dialect=dialect)

    except Exception:
        # If parsing/transformation fails, fall back to simple string append
        # This maintains backward compatibility
        sql_upper = sql.strip().upper()
        if "LIMIT" not in sql_upper:
            return f"{sql.rstrip(';')} LIMIT {limit};"
        return sql
