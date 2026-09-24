"""Guarded database execution shared by agent and protocol adapters."""

from dataclasses import dataclass
from typing import Any

import sqlglot
from sqlglot import exp

from sqlsaber.database.base import BaseDatabaseConnection

from .sql_guard import add_limit, validate_sql


class InvalidQuery(ValueError):
    """A query rejected before reaching the database."""


@dataclass
class ExecutionResult:
    rows: list[dict[str, Any]]
    query_type: str
    auto_limit_applied: bool
    truncated: bool = False


async def execute_guarded(
    connection: BaseDatabaseConnection,
    query: str,
    *,
    allow_dangerous: bool = False,
    max_rows: int = 1000,
    bounded: bool = False,
) -> ExecutionResult:
    """Validate and execute SQL, optionally bounding the complete read result.

    The agent retains its historical automatic-limit behavior. Protocol callers
    use bounded mode, which caps the top-level limit and reads one extra row to
    distinguish a complete result from a truncated one. Inner limits, ordering,
    offsets, and column names remain unchanged.
    """
    if not query:
        raise InvalidQuery("No query provided")
    dialect = connection.sqlglot_dialect
    validation = validate_sql(query, dialect, allow_dangerous=allow_dangerous)
    if not validation.allowed:
        raise InvalidQuery(validation.reason or "Query rejected")
    auto_limit = bool(validation.is_select and max_rows and not validation.has_limit)
    if bounded:
        if allow_dangerous or max_rows < 1 or not validation.is_select:
            raise ValueError(
                "Bounded execution requires a read-only query and row limit"
            )
        statement = sqlglot.parse_one(query, read=dialect)
        if not isinstance(statement, exp.Query):
            raise InvalidQuery("Cannot safely bound this query")
        row_limit = max_rows + 1
        limit = statement.args.get("limit")
        if limit is not None:
            count = (
                limit.args.get("count")
                if isinstance(limit, exp.Fetch)
                else limit.expression
            )
            if not isinstance(count, exp.Literal) or not count.is_int:
                raise InvalidQuery("Use a nonnegative integer LIMIT")
            if limit.args.get("limit_options"):
                raise InvalidQuery("LIMIT options are not supported")
            row_limit = min(row_limit, int(count.this))
        query = statement.limit(row_limit).sql(dialect=dialect)
    elif auto_limit:
        query = add_limit(query, dialect, max_rows)
    query_type = validation.query_type or "other"
    rows = await connection.execute_query(
        query,
        commit=bool(allow_dangerous and query_type in {"dml", "ddl"}),
        read_only=not allow_dangerous,
    )
    truncated = bounded and len(rows) > max_rows
    return ExecutionResult(
        rows[:max_rows] if truncated else rows, query_type, auto_limit, truncated
    )
