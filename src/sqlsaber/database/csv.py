"""CSV database connection using DuckDB backend."""

import asyncio
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import duckdb

from .base import DEFAULT_QUERY_TIMEOUT, BaseDatabaseConnection, QueryTimeoutError
from .duckdb import DuckDBSchemaIntrospector


def _execute_duckdb_transaction(
    conn: duckdb.DuckDBPyConnection,
    query: str,
    args: tuple[Any, ...],
    commit: bool = False,
) -> list[dict[str, Any]]:
    """Run a DuckDB query inside a transaction and return list of dicts.

    If commit=True, commits on success instead of rolling back.
    """
    conn.execute("BEGIN TRANSACTION")
    success = False
    try:
        if args:
            conn.execute(query, args)
        else:
            conn.execute(query)

        if conn.description is None:
            rows: list[dict[str, Any]] = []
        else:
            columns = [col[0] for col in conn.description]
            data = conn.fetchall()
            rows = [dict(zip(columns, row)) for row in data]

        success = True
        return rows
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        if success:
            if commit:
                conn.execute("COMMIT")
            else:
                conn.execute("ROLLBACK")


class CSVConnection(BaseDatabaseConnection):
    """CSV file connection using DuckDB per query."""

    def __init__(self, connection_string: str):
        super().__init__(connection_string)

        raw_path = connection_string.replace("csv:///", "", 1)
        self.csv_path = raw_path.split("?", 1)[0]

        self.delimiter = ","
        self.encoding = "utf-8"
        self.has_header = True

        parsed = urlparse(connection_string)
        if parsed.query:
            params = parse_qs(parsed.query)
            self.delimiter = params.get("delimiter", [self.delimiter])[0]
            self.encoding = params.get("encoding", [self.encoding])[0]
            self.has_header = params.get("header", ["true"])[0].lower() == "true"

        self.table_name = Path(self.csv_path).stem or "csv_table"

    @property
    def sqlglot_dialect(self) -> str:
        """Return the sqlglot dialect name."""
        return "duckdb"

    @property
    def display_name(self) -> str:
        """Return the human-readable name."""
        return "DuckDB"

    async def get_pool(self):
        """CSV connections do not maintain a pool."""
        return None

    async def close(self):
        """No persistent resources to close for CSV connections."""
        pass

    def _quote_identifier(self, identifier: str) -> str:
        escaped = identifier.replace('"', '""')
        return f'"{escaped}"'

    def _quote_literal(self, value: str) -> str:
        escaped = value.replace("'", "''")
        return f"'{escaped}'"

    def _normalized_encoding(self) -> str | None:
        encoding = (self.encoding or "").strip()
        if not encoding or encoding.lower() == "utf-8":
            return None
        return encoding.replace("-", "").replace("_", "").upper()

    def _create_view(self, conn: duckdb.DuckDBPyConnection) -> None:
        header_literal = "TRUE" if self.has_header else "FALSE"
        option_parts = [f"HEADER={header_literal}"]

        if self.delimiter:
            option_parts.append(f"DELIM={self._quote_literal(self.delimiter)}")

        encoding = self._normalized_encoding()
        if encoding:
            option_parts.append(f"ENCODING={self._quote_literal(encoding)}")

        options_sql = ""
        if option_parts:
            options_sql = ", " + ", ".join(option_parts)

        base_relation_sql = (
            f"read_csv_auto({self._quote_literal(self.csv_path)}{options_sql})"
        )

        create_view_sql = (
            f"CREATE VIEW {self._quote_identifier(self.table_name)} AS "
            f"SELECT * FROM {base_relation_sql}"
        )
        conn.execute(create_view_sql)

    async def execute_query(
        self, query: str, *args, timeout: float | None = None, commit: bool = False
    ) -> list[dict[str, Any]]:
        effective_timeout = timeout or DEFAULT_QUERY_TIMEOUT
        args_tuple = tuple(args) if args else tuple()

        def _run_query() -> list[dict[str, Any]]:
            conn = duckdb.connect(":memory:")
            try:
                self._create_view(conn)
                return _execute_duckdb_transaction(conn, query, args_tuple, commit)
            finally:
                conn.close()

        try:
            return await asyncio.wait_for(
                asyncio.to_thread(_run_query), timeout=effective_timeout
            )
        except asyncio.TimeoutError as exc:
            raise QueryTimeoutError(effective_timeout or 0) from exc


class CSVSchemaIntrospector(DuckDBSchemaIntrospector):
    """CSV-specific schema introspection using DuckDB backend."""

    pass
