"""SQLite database connection and schema introspection."""

import asyncio
from typing import Any

import aiosqlite

from .base import (
    DEFAULT_QUERY_TIMEOUT,
    BaseDatabaseConnection,
    BaseSchemaIntrospector,
    QueryTimeoutError,
)


class SQLiteConnection(BaseDatabaseConnection):
    """SQLite database connection using aiosqlite."""

    def __init__(self, connection_string: str):
        super().__init__(connection_string)
        # Extract database path from sqlite:///path format
        self.database_path = connection_string.replace("sqlite:///", "")

    @property
    def sqlglot_dialect(self) -> str:
        """Return the sqlglot dialect name."""
        return "sqlite"

    @property
    def display_name(self) -> str:
        """Return the human-readable name."""
        return "SQLite"

    async def get_pool(self):
        """SQLite doesn't use connection pooling, return database path."""
        return self.database_path

    async def close(self):
        """SQLite connections are created per query, no persistent pool to close."""
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

        async with aiosqlite.connect(self.database_path) as conn:
            conn.row_factory = aiosqlite.Row

            await conn.execute("BEGIN")
            success = False
            try:
                # Execute query with client-side timeout (SQLite has no server-side timeout)
                if effective_timeout:
                    cursor = await asyncio.wait_for(
                        conn.execute(query, args if args else ()),
                        timeout=effective_timeout,
                    )
                    rows = await asyncio.wait_for(
                        cursor.fetchall(), timeout=effective_timeout
                    )
                else:
                    cursor = await conn.execute(query, args if args else ())
                    rows = await cursor.fetchall()

                success = True
                return [dict(row) for row in rows]
            except asyncio.TimeoutError as exc:
                raise QueryTimeoutError(effective_timeout or 0) from exc
            finally:
                if success and commit:
                    await conn.commit()
                else:
                    await conn.rollback()


class SQLiteSchemaIntrospector(BaseSchemaIntrospector):
    """SQLite-specific schema introspection."""

    async def _execute_query(
        self, connection, query: str, params: tuple[str, ...] = ()
    ) -> list[dict[str, Any]]:
        """Helper method to execute queries on both SQLite and CSV connections."""
        # Handle both SQLite and CSV connections
        if hasattr(connection, "database_path"):
            # Regular SQLite connection
            async with aiosqlite.connect(connection.database_path) as conn:
                conn.row_factory = aiosqlite.Row
                cursor = await conn.execute(query, params)
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
        else:
            # CSV connection - use the existing connection
            conn = await connection.get_pool()
            cursor = await conn.execute(query, params)
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def get_tables_info(
        self, connection, table_pattern: str | None = None
    ) -> list[dict[str, Any]]:
        """Get tables information for SQLite.

        Note: SQLite does not support native table comments, so table_comment is always None.
        """
        where_conditions = ["type IN ('table', 'view')", "name NOT LIKE 'sqlite_%'"]
        params: tuple[str, ...] = ()

        if table_pattern:
            where_conditions.append("name LIKE ?")
            params = (table_pattern,)

        query = f"""
            SELECT
                'main' as table_schema,
                name as table_name,
                type as table_type,
                NULL as table_comment
            FROM sqlite_master
            WHERE {" AND ".join(where_conditions)}
            ORDER BY name;
        """

        return await self._execute_query(connection, query, params)

    async def get_columns_info(
        self, connection, tables: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Get columns information for SQLite.

        Note: SQLite does not support native column comments, so column_comment is always None.
        """
        if not tables:
            return []

        columns = []
        for table in tables:
            table_name = table["table_name"]

            # Get table info using PRAGMA
            pragma_query = f"PRAGMA table_info({table_name})"
            table_columns = await self._execute_query(connection, pragma_query)

            for col in table_columns:
                columns.append(
                    {
                        "table_schema": "main",
                        "table_name": table_name,
                        "column_name": col["name"],
                        "data_type": col["type"],
                        "is_nullable": "YES" if not col["notnull"] else "NO",
                        "column_default": col["dflt_value"],
                        "character_maximum_length": None,
                        "numeric_precision": None,
                        "numeric_scale": None,
                        "column_comment": None,
                    }
                )

        return columns

    async def get_foreign_keys_info(
        self, connection, tables: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Get foreign keys information for SQLite."""
        if not tables:
            return []

        foreign_keys = []
        for table in tables:
            table_name = table["table_name"]

            # Get foreign key info using PRAGMA
            pragma_query = f"PRAGMA foreign_key_list({table_name})"
            table_fks = await self._execute_query(connection, pragma_query)

            for fk in table_fks:
                foreign_keys.append(
                    {
                        "table_schema": "main",
                        "table_name": table_name,
                        "column_name": fk["from"],
                        "foreign_table_schema": "main",
                        "foreign_table_name": fk["table"],
                        "foreign_column_name": fk["to"],
                    }
                )

        return foreign_keys

    async def get_primary_keys_info(
        self, connection, tables: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Get primary keys information for SQLite."""
        if not tables:
            return []

        primary_keys = []
        for table in tables:
            table_name = table["table_name"]

            # Get table info using PRAGMA to find primary keys
            pragma_query = f"PRAGMA table_info({table_name})"
            table_columns = await self._execute_query(connection, pragma_query)

            for col in table_columns:
                if col["pk"]:  # Primary key indicator
                    primary_keys.append(
                        {
                            "table_schema": "main",
                            "table_name": table_name,
                            "column_name": col["name"],
                        }
                    )

        return primary_keys

    async def get_indexes_info(
        self, connection, tables: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Get indexes information for SQLite."""
        if not tables:
            return []

        indexes = []
        for table in tables:
            table_name = table["table_name"]

            # Get index list using PRAGMA
            pragma_query = f"PRAGMA index_list({table_name})"
            table_indexes = await self._execute_query(connection, pragma_query)

            for idx in table_indexes:
                idx_name = idx["name"]
                unique = bool(idx["unique"])

                # Skip auto-generated primary key indexes
                if idx_name.startswith("sqlite_autoindex_"):
                    continue

                # Get index columns using PRAGMA
                pragma_info_query = f"PRAGMA index_info({idx_name})"
                idx_cols = await self._execute_query(connection, pragma_info_query)
                columns = [
                    c["name"] for c in sorted(idx_cols, key=lambda r: r["seqno"])
                ]

                indexes.append(
                    {
                        "table_schema": "main",
                        "table_name": table_name,
                        "index_name": idx_name,
                        "is_unique": unique,
                        "index_type": None,  # SQLite only has B-tree currently
                        "column_names": columns,
                    }
                )

        return indexes

    async def list_tables_info(self, connection) -> list[dict[str, Any]]:
        """Get list of tables with basic information for SQLite.

        Note: SQLite does not support native table comments, so table_comment is always None.
        """
        # Get table names without row counts for better performance
        tables_query = """
            SELECT
                'main' as table_schema,
                name as table_name,
                type as table_type,
                NULL as table_comment
            FROM sqlite_master
            WHERE type IN ('table', 'view')
            AND name NOT LIKE 'sqlite_%'
            ORDER BY name;
        """

        tables = await self._execute_query(connection, tables_query)

        # Convert to expected format
        return [
            {
                "table_schema": table["table_schema"],
                "table_name": table["table_name"],
                "table_type": table["table_type"],
                "table_comment": table["table_comment"],
            }
            for table in tables
        ]
