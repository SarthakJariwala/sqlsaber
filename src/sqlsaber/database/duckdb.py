"""DuckDB database connection and schema introspection."""

import asyncio
from typing import Any

import duckdb

from .base import (
    DEFAULT_QUERY_TIMEOUT,
    BaseDatabaseConnection,
    BaseSchemaIntrospector,
    QueryTimeoutError,
)


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


class DuckDBConnection(BaseDatabaseConnection):
    """DuckDB database connection using duckdb Python API."""

    def __init__(self, connection_string: str):
        super().__init__(connection_string)
        if connection_string.startswith("duckdb:///"):
            db_path = connection_string.replace("duckdb:///", "", 1)
        elif connection_string.startswith("duckdb://"):
            db_path = connection_string.replace("duckdb://", "", 1)
        else:
            db_path = connection_string

        self.database_path = db_path or ":memory:"

    @property
    def sqlglot_dialect(self) -> str:
        """Return the sqlglot dialect name."""
        return "duckdb"

    @property
    def display_name(self) -> str:
        """Return the human-readable name."""
        return "DuckDB"

    async def get_pool(self):
        """DuckDB creates connections per query, return database path."""
        return self.database_path

    async def close(self):
        """DuckDB connections are created per query, no persistent pool to close."""
        pass

    async def execute_query(
        self, query: str, *args, timeout: float | None = None, commit: bool = False
    ) -> list[dict[str, Any]]:
        """Execute a query and return results as list of dicts.

        By default, all queries run in a transaction that is rolled back at the end,
        ensuring no changes are persisted to the database.

        If commit=True, the transaction will be committed on success.
        """
        effective_timeout = timeout or DEFAULT_QUERY_TIMEOUT

        args_tuple = tuple(args) if args else tuple()

        def _run_query() -> list[dict[str, Any]]:
            conn = duckdb.connect(self.database_path)
            try:
                return _execute_duckdb_transaction(conn, query, args_tuple, commit)
            finally:
                conn.close()

        try:
            return await asyncio.wait_for(
                asyncio.to_thread(_run_query), timeout=effective_timeout
            )
        except asyncio.TimeoutError as exc:
            raise QueryTimeoutError(effective_timeout or 0) from exc


class DuckDBSchemaIntrospector(BaseSchemaIntrospector):
    """DuckDB-specific schema introspection."""

    def _get_excluded_schemas(self, connection) -> list[str]:
        """Return schemas to exclude during introspection."""
        defaults = ["information_schema", "pg_catalog", "duckdb_catalog"]
        return self._merge_excluded_schemas(
            connection, defaults, env_var="SQLSABER_DUCKDB_EXCLUDE_SCHEMAS"
        )

    async def _execute_query(
        self,
        connection,
        query: str,
        params: tuple[Any, ...] = (),
    ) -> list[dict[str, Any]]:
        """Run a DuckDB query on a thread and return list of dictionaries."""

        params_tuple = tuple(params)

        def fetch_rows(conn: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
            cursor = conn.execute(query, params_tuple)
            if cursor.description is None:
                return []

            columns = [col[0] for col in cursor.description]
            rows = conn.fetchall()
            return [dict(zip(columns, row)) for row in rows]

        # Handle CSV connections differently
        if hasattr(connection, "execute_query") and (
            hasattr(connection, "csv_path") or hasattr(connection, "csv_sources")
        ):
            return await connection.execute_query(query, *params_tuple)

        def run_query() -> list[dict[str, Any]]:
            conn = duckdb.connect(connection.database_path)
            try:
                return fetch_rows(conn)
            finally:
                conn.close()

        return await asyncio.to_thread(run_query)

    async def get_tables_info(
        self, connection, table_pattern: str | None = None
    ) -> list[dict[str, Any]]:
        """Get tables information for DuckDB."""
        excluded = self._get_excluded_schemas(connection)
        where_conditions: list[str] = []
        params: list[Any] = []

        if excluded:
            placeholders = ", ".join(["?"] * len(excluded))
            where_conditions.append(f"t.table_schema NOT IN ({placeholders})")
            params.extend(excluded)

        if table_pattern:
            if "." in table_pattern:
                schema_pattern, table_name_pattern = table_pattern.split(".", 1)
                where_conditions.append(
                    "(t.table_schema LIKE ? AND t.table_name LIKE ?)"
                )
                params.extend([schema_pattern, table_name_pattern])
            else:
                where_conditions.append(
                    "(t.table_name LIKE ? OR t.table_schema || '.' || t.table_name LIKE ?)"
                )
                params.extend([table_pattern, table_pattern])

        if not where_conditions:
            where_conditions.append("1=1")

        query = f"""
            SELECT
                t.table_schema,
                t.table_name,
                t.table_type,
                dt.comment AS table_comment
            FROM information_schema.tables t
            LEFT JOIN duckdb_tables() dt
                ON t.table_schema = dt.schema_name
                AND t.table_name = dt.table_name
            WHERE {" AND ".join(where_conditions)}
            ORDER BY t.table_schema, t.table_name;
        """

        return await self._execute_query(connection, query, tuple(params))

    async def get_columns_info(
        self, connection, tables: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Get columns information for DuckDB."""
        if not tables:
            return []

        table_filters = []
        for table in tables:
            table_filters.append("(c.table_schema = ? AND c.table_name = ?)")

        params: list[Any] = []
        for table in tables:
            params.extend([table["table_schema"], table["table_name"]])

        query = f"""
            SELECT
                c.table_schema,
                c.table_name,
                c.column_name,
                c.data_type,
                c.is_nullable,
                c.column_default,
                c.character_maximum_length,
                c.numeric_precision,
                c.numeric_scale,
                dc.comment AS column_comment
            FROM information_schema.columns c
            LEFT JOIN duckdb_columns() dc
                ON c.table_schema = dc.schema_name
                AND c.table_name = dc.table_name
                AND c.column_name = dc.column_name
            WHERE {" OR ".join(table_filters)}
            ORDER BY c.table_schema, c.table_name, c.ordinal_position;
        """

        return await self._execute_query(connection, query, tuple(params))

    async def get_foreign_keys_info(
        self, connection, tables: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Get foreign keys information for DuckDB."""
        if not tables:
            return []

        table_filters = []
        params: list[Any] = []
        for table in tables:
            table_filters.append("(kcu.table_schema = ? AND kcu.table_name = ?)")
            params.extend([table["table_schema"], table["table_name"]])

        query = f"""
            SELECT
                kcu.table_schema,
                kcu.table_name,
                kcu.column_name,
                ccu.table_schema AS foreign_table_schema,
                ccu.table_name AS foreign_table_name,
                ccu.column_name AS foreign_column_name
            FROM information_schema.referential_constraints AS rc
            JOIN information_schema.key_column_usage AS kcu
                ON rc.constraint_schema = kcu.constraint_schema
                AND rc.constraint_name = kcu.constraint_name
            JOIN information_schema.key_column_usage AS ccu
                ON rc.unique_constraint_schema = ccu.constraint_schema
                AND rc.unique_constraint_name = ccu.constraint_name
                AND ccu.ordinal_position = kcu.position_in_unique_constraint
            WHERE {" OR ".join(table_filters)}
            ORDER BY kcu.table_schema, kcu.table_name, kcu.ordinal_position;
        """

        return await self._execute_query(connection, query, tuple(params))

    async def get_primary_keys_info(
        self, connection, tables: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Get primary keys information for DuckDB."""
        if not tables:
            return []

        table_filters = []
        params: list[Any] = []
        for table in tables:
            table_filters.append("(tc.table_schema = ? AND tc.table_name = ?)")
            params.extend([table["table_schema"], table["table_name"]])

        query = f"""
            SELECT
                tc.table_schema,
                tc.table_name,
                kcu.column_name
            FROM information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
                ON tc.constraint_name = kcu.constraint_name
                AND tc.constraint_schema = kcu.constraint_schema
            WHERE tc.constraint_type = 'PRIMARY KEY'
                AND ({" OR ".join(table_filters)})
            ORDER BY tc.table_schema, tc.table_name, kcu.ordinal_position;
        """

        return await self._execute_query(connection, query, tuple(params))

    async def get_indexes_info(
        self, connection, tables: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Get indexes information for DuckDB."""
        if not tables:
            return []

        indexes: list[dict[str, Any]] = []
        for table in tables:
            schema = table["table_schema"]
            table_name = table["table_name"]
            query = """
                SELECT
                    schema_name,
                    table_name,
                    index_name,
                    sql
                FROM duckdb_indexes()
                WHERE schema_name = ? AND table_name = ?;
            """
            rows = await self._execute_query(connection, query, (schema, table_name))

            for row in rows:
                sql_text = (row.get("sql") or "").strip()
                upper_sql = sql_text.upper()
                unique = "UNIQUE" in upper_sql.split("(")[0]

                columns: list[str] = []
                if "(" in sql_text and ")" in sql_text:
                    column_section = sql_text[
                        sql_text.find("(") + 1 : sql_text.rfind(")")
                    ]
                    columns = [
                        col.strip().strip('"')
                        for col in column_section.split(",")
                        if col.strip()
                    ]

                indexes.append(
                    {
                        "table_schema": row.get("schema_name") or schema or "main",
                        "table_name": row.get("table_name") or table_name,
                        "index_name": row.get("index_name"),
                        "is_unique": unique,
                        "index_type": None,
                        "column_names": columns,
                    }
                )

        return indexes

    async def list_tables_info(self, connection) -> list[dict[str, Any]]:
        """Get list of tables with basic information for DuckDB."""
        excluded = self._get_excluded_schemas(connection)
        params: list[Any] = []
        if excluded:
            placeholders = ", ".join(["?"] * len(excluded))
            where_clause = f"WHERE t.table_schema NOT IN ({placeholders})"
            params.extend(excluded)
        else:
            where_clause = ""

        query = f"""
            SELECT
                t.table_schema,
                t.table_name,
                t.table_type,
                dt.comment AS table_comment
            FROM information_schema.tables t
            LEFT JOIN duckdb_tables() dt
                ON t.table_schema = dt.schema_name
                AND t.table_name = dt.table_name
            {where_clause}
            ORDER BY t.table_schema, t.table_name;
        """

        return await self._execute_query(connection, query, tuple(params))
