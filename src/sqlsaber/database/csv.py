"""CSV and Parquet file connections using the same DuckDB backend."""

import asyncio
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import duckdb

from .base import DEFAULT_QUERY_TIMEOUT, BaseDatabaseConnection, QueryTimeoutError
from .duckdb import (
    _duckdb_interrupt_timer,
    _execute_duckdb_transaction,
    apply_read_only_lockdown,
)

__all__ = ["CSVConnection"]


class CSVConnection(BaseDatabaseConnection):
    """Local CSV or Parquet connection using DuckDB per query.

    The historical class and attribute names are retained for compatibility.
    """

    def __init__(self, connection_string: str):
        super().__init__(connection_string)

        self.file_format = urlparse(connection_string).scheme
        if self.file_format not in {"csv", "parquet"}:
            raise ValueError("Expected a csv:/// or parquet:/// connection string")
        raw_path = connection_string.removeprefix(f"{self.file_format}:///")
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

    def _create_table(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Materialize the source file into a real table.

        A materialized table (rather than a view over a file reader) lets the
        session lock down external file access afterwards while still serving the
        intended data — a view would re-read the file on every query and break
        under the lockdown.
        """
        if self.file_format == "parquet":
            conn.execute(
                f"CREATE TABLE {self._quote_identifier(self.table_name)} AS "
                f"SELECT * FROM read_parquet({self._quote_literal(self.csv_path)})"
            )
            return

        header_literal = "TRUE" if self.has_header else "FALSE"
        option_parts: list[str] = [f"HEADER={header_literal}"]

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

        create_table_sql = (
            f"CREATE TABLE {self._quote_identifier(self.table_name)} AS "
            f"SELECT * FROM {base_relation_sql}"
        )
        conn.execute(create_table_sql)

    async def execute_query(
        self,
        query: str,
        *args,
        timeout: float | None = None,
        commit: bool = False,
        read_only: bool = False,
    ) -> list[dict[str, Any]]:
        effective_timeout = timeout or DEFAULT_QUERY_TIMEOUT
        args_tuple = tuple(args) if args else tuple()

        def _run_query() -> list[dict[str, Any]]:
            conn = duckdb.connect(":memory:")
            try:
                with _duckdb_interrupt_timer(conn, effective_timeout):
                    self._create_table(conn)
                    if read_only:
                        apply_read_only_lockdown(conn)
                    return _execute_duckdb_transaction(
                        conn, query, args_tuple, commit, timeout=effective_timeout
                    )
            except duckdb.InterruptException as exc:
                raise QueryTimeoutError(effective_timeout or 0) from exc
            finally:
                conn.close()

        try:
            wait_timeout = (
                effective_timeout + 5 if effective_timeout else effective_timeout
            )
            return await asyncio.wait_for(
                asyncio.to_thread(_run_query), timeout=wait_timeout
            )
        except asyncio.TimeoutError as exc:
            raise QueryTimeoutError(effective_timeout or 0) from exc
