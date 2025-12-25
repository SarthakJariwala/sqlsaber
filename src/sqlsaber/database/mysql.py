"""MySQL database connection and schema introspection."""

import asyncio
import ssl
from typing import Any
from urllib.parse import parse_qs, urlparse

import aiomysql

from .base import (
    DEFAULT_QUERY_TIMEOUT,
    BaseDatabaseConnection,
    BaseSchemaIntrospector,
    QueryTimeoutError,
)


class MySQLConnection(BaseDatabaseConnection):
    """MySQL database connection using aiomysql."""

    def __init__(self, connection_string: str):
        super().__init__(connection_string)
        self._pool: aiomysql.Pool | None = None
        self._parse_connection_string()

    @property
    def sqlglot_dialect(self) -> str:
        """Return the sqlglot dialect name."""
        return "mysql"

    @property
    def display_name(self) -> str:
        """Return the human-readable name."""
        return "MySQL"

    def _parse_connection_string(self):
        """Parse MySQL connection string into components."""
        parsed = urlparse(self.connection_string)
        self.host = parsed.hostname or "localhost"
        self.port = parsed.port or 3306
        self.database = parsed.path.lstrip("/") if parsed.path else ""
        self.user = parsed.username or ""
        self.password = parsed.password or ""

        # Parse SSL parameters
        self.ssl_params = {}
        if parsed.query:
            params = parse_qs(parsed.query)

            ssl_mode = params.get("ssl_mode", [None])[0]
            if ssl_mode:
                # Map SSL modes to aiomysql SSL parameters
                if ssl_mode.upper() == "DISABLED":
                    self.ssl_params["ssl"] = None
                elif ssl_mode.upper() in [
                    "PREFERRED",
                    "REQUIRED",
                    "VERIFY_CA",
                    "VERIFY_IDENTITY",
                ]:
                    ssl_context = ssl.create_default_context()

                    if ssl_mode.upper() == "REQUIRED":
                        ssl_context.check_hostname = False
                        ssl_context.verify_mode = ssl.CERT_NONE
                    elif ssl_mode.upper() == "VERIFY_CA":
                        ssl_context.check_hostname = False
                        ssl_context.verify_mode = ssl.CERT_REQUIRED
                    elif ssl_mode.upper() == "VERIFY_IDENTITY":
                        ssl_context.check_hostname = True
                        ssl_context.verify_mode = ssl.CERT_REQUIRED

                    # Load certificates if provided
                    ssl_ca = params.get("ssl_ca", [None])[0]
                    ssl_cert = params.get("ssl_cert", [None])[0]
                    ssl_key = params.get("ssl_key", [None])[0]

                    if ssl_ca:
                        ssl_context.load_verify_locations(ssl_ca)

                    if ssl_cert and ssl_key:
                        ssl_context.load_cert_chain(ssl_cert, ssl_key)

                    self.ssl_params["ssl"] = ssl_context

    async def get_pool(self) -> aiomysql.Pool:
        """Get or create connection pool."""
        if self._pool is None:
            pool_kwargs = {
                "host": self.host,
                "port": self.port,
                "user": self.user,
                "password": self.password,
                "db": self.database,
                "minsize": 1,
                "maxsize": 10,
                "autocommit": False,
            }

            # Add SSL parameters if configured
            pool_kwargs.update(self.ssl_params)

            self._pool = await aiomysql.create_pool(**pool_kwargs)
        return self._pool

    async def close(self):
        """Close the connection pool."""
        if self._pool:
            self._pool.close()
            await self._pool.wait_closed()
            self._pool = None

    async def execute_query(
        self, query: str, *args, timeout: float | None = None, commit: bool = False
    ) -> list[dict[str, Any]]:
        """Execute a query and return results as list of dicts.

        By default, all queries run in a transaction that is rolled back at the end,
        ensuring no changes are persisted to the database.

        If commit=True, the transaction will be committed on success.
        """
        effective_timeout = timeout or DEFAULT_QUERY_TIMEOUT
        pool = await self.get_pool()

        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                await conn.begin()
                success = False
                try:
                    # Set server-side timeout if specified
                    if effective_timeout:
                        # Clamp timeout to sane range (10ms to 5 minutes) and validate
                        timeout_ms = max(10, min(int(effective_timeout * 1000), 300000))
                        await cursor.execute(
                            f"SET SESSION MAX_EXECUTION_TIME = {timeout_ms}"
                        )

                    # Execute query with client-side timeout
                    if effective_timeout:
                        await asyncio.wait_for(
                            cursor.execute(query, args if args else None),
                            timeout=effective_timeout,
                        )
                        rows = await asyncio.wait_for(
                            cursor.fetchall(), timeout=effective_timeout
                        )
                    else:
                        await cursor.execute(query, args if args else None)
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


class MySQLSchemaIntrospector(BaseSchemaIntrospector):
    """MySQL-specific schema introspection."""

    def _get_excluded_schemas(self, connection) -> list[str]:
        """Return schemas to exclude during introspection."""
        defaults = ["information_schema", "performance_schema", "mysql", "sys"]
        return self._merge_excluded_schemas(
            connection, defaults, env_var="SQLSABER_MYSQL_EXCLUDE_SCHEMAS"
        )

    def _build_table_filter_clause(self, tables: list) -> tuple[str, list]:
        """Build row constructor with bind parameters for table filtering.

        Args:
            tables: List of table dictionaries with table_schema and table_name keys

        Returns:
            Tuple of (placeholders, params) for use in SQL queries
        """
        if not tables:
            return "", []

        table_pairs = [(table["table_schema"], table["table_name"]) for table in tables]
        placeholders = ", ".join(["(%s, %s)"] * len(table_pairs))
        params = [value for pair in table_pairs for value in pair]
        return placeholders, params

    async def get_tables_info(
        self, connection, table_pattern: str | None = None
    ) -> list[dict[str, Any]]:
        """Get tables information for MySQL."""
        pool = await connection.get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                # Build WHERE clause for filtering
                excluded = self._get_excluded_schemas(connection)
                where_conditions = []
                params: list[Any] = []

                if excluded:
                    placeholders = ", ".join(["%s"] * len(excluded))
                    where_conditions.append(f"table_schema NOT IN ({placeholders})")
                    params.extend(excluded)

                if table_pattern:
                    # Support patterns like 'schema.table' or just 'table'
                    if "." in table_pattern:
                        schema_pattern, table_name_pattern = table_pattern.split(".", 1)
                        where_conditions.append(
                            "(table_schema LIKE %s AND table_name LIKE %s)"
                        )
                        params.extend([schema_pattern, table_name_pattern])
                    else:
                        where_conditions.append(
                            "(table_name LIKE %s OR CONCAT(table_schema, '.', table_name) LIKE %s)"
                        )
                        params.extend([table_pattern, table_pattern])

                if not where_conditions:
                    where_conditions.append("1=1")

                # Get tables
                tables_query = f"""
                    SELECT
                        table_schema,
                        table_name,
                        table_type,
                        table_comment
                    FROM information_schema.tables
                    WHERE {" AND ".join(where_conditions)}
                    ORDER BY table_schema, table_name;
                """
                await cursor.execute(tables_query, params)
                return await cursor.fetchall()

    async def get_columns_info(
        self, connection, tables: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Get columns information for MySQL."""
        if not tables:
            return []

        pool = await connection.get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                placeholders, params = self._build_table_filter_clause(tables)

                columns_query = f"""
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
                        c.column_comment
                    FROM information_schema.columns c
                    WHERE (c.table_schema, c.table_name) IN ({placeholders})
                    ORDER BY c.table_schema, c.table_name, c.ordinal_position;
                """
                await cursor.execute(columns_query, params)
                return await cursor.fetchall()

    async def get_foreign_keys_info(
        self, connection, tables: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Get foreign keys information for MySQL."""
        if not tables:
            return []

        pool = await connection.get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                placeholders, params = self._build_table_filter_clause(tables)

                fk_query = f"""
                    SELECT
                        tc.table_schema,
                        tc.table_name,
                        kcu.column_name,
                        rc.unique_constraint_schema AS foreign_table_schema,
                        rc.referenced_table_name AS foreign_table_name,
                        kcu.referenced_column_name AS foreign_column_name
                    FROM information_schema.table_constraints AS tc
                    JOIN information_schema.key_column_usage AS kcu
                        ON tc.constraint_name = kcu.constraint_name
                        AND tc.table_schema = kcu.table_schema
                    JOIN information_schema.referential_constraints AS rc
                        ON tc.constraint_name = rc.constraint_name
                        AND tc.table_schema = rc.constraint_schema
                    WHERE tc.constraint_type = 'FOREIGN KEY'
                        AND (tc.table_schema, tc.table_name) IN ({placeholders});
                """
                await cursor.execute(fk_query, params)
                return await cursor.fetchall()

    async def get_primary_keys_info(
        self, connection, tables: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Get primary keys information for MySQL."""
        if not tables:
            return []

        pool = await connection.get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                placeholders, params = self._build_table_filter_clause(tables)

                pk_query = f"""
                    SELECT
                        tc.table_schema,
                        tc.table_name,
                        kcu.column_name
                    FROM information_schema.table_constraints AS tc
                    JOIN information_schema.key_column_usage AS kcu
                        ON tc.constraint_name = kcu.constraint_name
                        AND tc.table_schema = kcu.table_schema
                    WHERE tc.constraint_type = 'PRIMARY KEY'
                        AND (tc.table_schema, tc.table_name) IN ({placeholders})
                    ORDER BY tc.table_schema, tc.table_name, kcu.ordinal_position;
                """
                await cursor.execute(pk_query, params)
                return await cursor.fetchall()

    async def get_indexes_info(
        self, connection, tables: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Get indexes information for MySQL."""
        if not tables:
            return []

        pool = await connection.get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                placeholders, params = self._build_table_filter_clause(tables)

                idx_query = f"""
                    SELECT
                        TABLE_SCHEMA   AS table_schema,
                        TABLE_NAME     AS table_name,
                        INDEX_NAME     AS index_name,
                        (NON_UNIQUE = 0) AS is_unique,
                        INDEX_TYPE     AS index_type,
                        GROUP_CONCAT(COLUMN_NAME ORDER BY SEQ_IN_INDEX) AS column_names
                    FROM INFORMATION_SCHEMA.STATISTICS
                    WHERE (TABLE_SCHEMA, TABLE_NAME) IN ({placeholders})
                    GROUP BY table_schema, table_name, index_name, is_unique, index_type
                    ORDER BY table_schema, table_name, index_name;
                """
                await cursor.execute(idx_query, params)
                return await cursor.fetchall()

    async def list_tables_info(self, connection) -> list[dict[str, Any]]:
        """Get list of tables with basic information for MySQL."""
        pool = await connection.get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                # Get tables without row counts for better performance
                excluded = self._get_excluded_schemas(connection)
                params: list[Any] = []
                if excluded:
                    placeholders = ", ".join(["%s"] * len(excluded))
                    where_clause = f"WHERE t.table_schema NOT IN ({placeholders})"
                    params.extend(excluded)
                else:
                    where_clause = ""

                tables_query = f"""
                    SELECT
                        t.table_schema,
                        t.table_name,
                        t.table_type,
                        t.table_comment
                    FROM information_schema.tables t
                    {where_clause}
                    ORDER BY t.table_schema, t.table_name;
                """
                await cursor.execute(tables_query, params if params else None)
                rows = await cursor.fetchall()

                # Convert rows to dictionaries
                return [
                    {
                        "table_schema": row["table_schema"],
                        "table_name": row["table_name"],
                        "table_type": row["table_type"],
                        "table_comment": row["table_comment"],
                    }
                    for row in rows
                ]
