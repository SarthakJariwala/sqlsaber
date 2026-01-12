"""Multiple-CSV database connection using DuckDB backend.

This connection creates one DuckDB view per CSV file for each query.
"""

import asyncio
from collections import Counter
from typing import Any
from urllib.parse import parse_qs, urlparse

import duckdb

from .base import DEFAULT_QUERY_TIMEOUT, BaseDatabaseConnection, QueryTimeoutError
from .csv import CSVConnection, _execute_duckdb_transaction


class CSVsConnection(BaseDatabaseConnection):
    """Connection that exposes multiple CSV files as DuckDB views."""

    def __init__(self, connection_string: str):
        super().__init__(connection_string)

        parsed = urlparse(connection_string)
        if parsed.scheme != "csvs":
            raise ValueError(
                f"Invalid connection string for CSVsConnection: {connection_string}"
            )

        params = parse_qs(parsed.query)
        specs = [spec for spec in params.get("spec", []) if spec.strip()]
        if not specs:
            raise ValueError(
                "csvs connection string must include one or more 'spec=' parameters"
            )

        sources = [CSVConnection(spec) for spec in specs]
        self._dedupe_view_names(sources)

        # Used by DuckDBSchemaIntrospector to route through execute_query.
        self.csv_sources: list[CSVConnection] = sources

    @staticmethod
    def _dedupe_view_names(sources: list[CSVConnection]) -> None:
        counts = Counter(src.table_name for src in sources)
        used: dict[str, int] = {}

        for src in sources:
            base = src.table_name
            if counts[base] <= 1:
                continue

            used[base] = used.get(base, 0) + 1
            if used[base] == 1:
                continue
            src.table_name = f"{base}_{used[base]}"

    @property
    def sqlglot_dialect(self) -> str:
        return "duckdb"

    @property
    def display_name(self) -> str:
        return "DuckDB"

    async def get_pool(self):
        return None

    async def close(self):
        pass

    async def execute_query(
        self, query: str, *args, timeout: float | None = None, commit: bool = False
    ) -> list[dict[str, Any]]:
        effective_timeout = timeout or DEFAULT_QUERY_TIMEOUT
        args_tuple = tuple(args) if args else tuple()

        def _run_query() -> list[dict[str, Any]]:
            conn = duckdb.connect(":memory:")
            try:
                for src in self.csv_sources:
                    src._create_view(conn)
                return _execute_duckdb_transaction(conn, query, args_tuple, commit)
            finally:
                conn.close()

        try:
            return await asyncio.wait_for(
                asyncio.to_thread(_run_query), timeout=effective_timeout
            )
        except asyncio.TimeoutError as exc:
            raise QueryTimeoutError(effective_timeout or 0) from exc
