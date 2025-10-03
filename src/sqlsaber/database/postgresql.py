"""PostgreSQL database connection and schema introspection."""

import asyncio
import ssl
from typing import Any
from urllib.parse import parse_qs, urlparse

import asyncpg

from .base import (
    DEFAULT_QUERY_TIMEOUT,
    BaseDatabaseConnection,
    BaseSchemaIntrospector,
    QueryTimeoutError,
)


class PostgreSQLConnection(BaseDatabaseConnection):
    """PostgreSQL database connection using asyncpg."""

    def __init__(self, connection_string: str):
        super().__init__(connection_string)
        self._pool: asyncpg.Pool | None = None
        self._ssl_context = self._create_ssl_context()

    @property
    def sqlglot_dialect(self) -> str:
        """Return the sqlglot dialect name."""
        return "postgres"

    def _create_ssl_context(self) -> ssl.SSLContext | None:
        """Create SSL context from connection string parameters."""
        parsed = urlparse(self.connection_string)
        if not parsed.query:
            return None

        params = parse_qs(parsed.query)
        ssl_mode = params.get("sslmode", [None])[0]

        if not ssl_mode or ssl_mode == "disable":
            return None

        # Create SSL context based on mode
        if ssl_mode in ["require", "verify-ca", "verify-full"]:
            ssl_context = ssl.create_default_context()

            # Configure certificate verification
            if ssl_mode == "require":
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
            elif ssl_mode == "verify-ca":
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_REQUIRED
            elif ssl_mode == "verify-full":
                ssl_context.check_hostname = True
                ssl_context.verify_mode = ssl.CERT_REQUIRED

            # Load certificates if provided
            ssl_ca = params.get("sslrootcert", [None])[0]
            ssl_cert = params.get("sslcert", [None])[0]
            ssl_key = params.get("sslkey", [None])[0]

            if ssl_ca:
                ssl_context.load_verify_locations(ssl_ca)

            if ssl_cert and ssl_key:
                ssl_context.load_cert_chain(ssl_cert, ssl_key)

            return ssl_context

        return None

    async def get_pool(self) -> asyncpg.Pool:
        """Get or create connection pool."""
        if self._pool is None:
            # Create pool with SSL context if configured
            if self._ssl_context:
                self._pool = await asyncpg.create_pool(
                    self.connection_string,
                    min_size=1,
                    max_size=10,
                    ssl=self._ssl_context,
                )
            else:
                self._pool = await asyncpg.create_pool(
                    self.connection_string, min_size=1, max_size=10
                )
        return self._pool

    async def close(self):
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None

    async def execute_query(
        self, query: str, *args, timeout: float | None = None
    ) -> list[dict[str, Any]]:
        """Execute a query and return results as list of dicts.

        All queries run in a transaction that is rolled back at the end,
        ensuring no changes are persisted to the database.
        """
        effective_timeout = timeout or DEFAULT_QUERY_TIMEOUT
        pool = await self.get_pool()

        async with pool.acquire() as conn:
            # Start a transaction that we'll always rollback
            transaction = conn.transaction()
            await transaction.start()

            try:
                # Set server-side timeout if specified
                if effective_timeout:
                    # Clamp timeout to sane range (10ms to 5 minutes) and validate
                    timeout_ms = max(10, min(int(effective_timeout * 1000), 300000))
                    await conn.execute(f"SET LOCAL statement_timeout = {timeout_ms}")

                # Execute query with client-side timeout
                if effective_timeout:
                    rows = await asyncio.wait_for(
                        conn.fetch(query, *args), timeout=effective_timeout
                    )
                else:
                    rows = await conn.fetch(query, *args)

                return [dict(row) for row in rows]
            except asyncio.TimeoutError as exc:
                raise QueryTimeoutError(effective_timeout or 0) from exc
            finally:
                # Always rollback to ensure no changes are committed
                await transaction.rollback()


class PostgreSQLSchemaIntrospector(BaseSchemaIntrospector):
    """PostgreSQL-specific schema introspection."""

    def _build_table_filter_clause(self, tables: list) -> tuple[str, list]:
        """Build VALUES clause with bind parameters for table filtering.

        Args:
            tables: List of table dictionaries with table_schema and table_name keys

        Returns:
            Tuple of (values_clause, params) for use in SQL queries
        """
        if not tables:
            return "", []

        table_pairs = [(table["table_schema"], table["table_name"]) for table in tables]
        values_clause = ", ".join(
            [f"(${2 * i + 1}, ${2 * i + 2})" for i in range(len(table_pairs))]
        )
        params = [value for pair in table_pairs for value in pair]
        return values_clause, params

    async def get_tables_info(
        self, connection, table_pattern: str | None = None
    ) -> dict[str, Any]:
        """Get tables information for PostgreSQL."""
        pool = await connection.get_pool()
        async with pool.acquire() as conn:
            # Build WHERE clause for filtering
            where_conditions = [
                "table_schema NOT IN ('pg_catalog', 'information_schema')"
            ]
            params = []

            if table_pattern:
                # Support patterns like 'schema.table' or just 'table'
                if "." in table_pattern:
                    schema_pattern, table_name_pattern = table_pattern.split(".", 1)
                    where_conditions.append(
                        "(table_schema LIKE $1 AND table_name LIKE $2)"
                    )
                    params.extend([schema_pattern, table_name_pattern])
                else:
                    where_conditions.append(
                        "(table_name LIKE $1 OR table_schema || '.' || table_name LIKE $1)"
                    )
                    params.append(table_pattern)

            # Get tables
            tables_query = f"""
                SELECT
                    table_schema,
                    table_name,
                    table_type
                FROM information_schema.tables
                WHERE {" AND ".join(where_conditions)}
                ORDER BY table_schema, table_name;
            """
            return await conn.fetch(tables_query, *params)

    async def get_columns_info(self, connection, tables: list) -> list:
        """Get columns information for PostgreSQL."""
        if not tables:
            return []

        pool = await connection.get_pool()
        async with pool.acquire() as conn:
            values_clause, params = self._build_table_filter_clause(tables)

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
                    c.numeric_scale
                FROM information_schema.columns c
                WHERE (c.table_schema, c.table_name) IN (VALUES {values_clause})
                ORDER BY c.table_schema, c.table_name, c.ordinal_position;
            """
            return await conn.fetch(columns_query, *params)

    async def get_foreign_keys_info(self, connection, tables: list) -> list:
        """Get foreign keys information for PostgreSQL."""
        if not tables:
            return []

        pool = await connection.get_pool()
        async with pool.acquire() as conn:
            values_clause, params = self._build_table_filter_clause(tables)

            fk_query = f"""
                WITH t(schema, name) AS (VALUES {values_clause})
                SELECT
                    tc.table_schema,
                    tc.table_name,
                    kcu.column_name,
                    ccu.table_schema AS foreign_table_schema,
                    ccu.table_name AS foreign_table_name,
                    ccu.column_name AS foreign_column_name
                FROM information_schema.table_constraints AS tc
                JOIN information_schema.key_column_usage AS kcu
                    ON tc.constraint_name = kcu.constraint_name
                    AND tc.table_schema = kcu.table_schema
                JOIN information_schema.constraint_column_usage AS ccu
                    ON ccu.constraint_name = tc.constraint_name
                    AND ccu.table_schema = tc.table_schema
                JOIN t ON t.schema = tc.table_schema AND t.name = tc.table_name
                WHERE tc.constraint_type = 'FOREIGN KEY';
            """
            return await conn.fetch(fk_query, *params)

    async def get_primary_keys_info(self, connection, tables: list) -> list:
        """Get primary keys information for PostgreSQL."""
        if not tables:
            return []

        pool = await connection.get_pool()
        async with pool.acquire() as conn:
            values_clause, params = self._build_table_filter_clause(tables)

            pk_query = f"""
                WITH t(schema, name) AS (VALUES {values_clause})
                SELECT
                    tc.table_schema,
                    tc.table_name,
                    kcu.column_name
                FROM information_schema.table_constraints AS tc
                JOIN information_schema.key_column_usage AS kcu
                    ON tc.constraint_name = kcu.constraint_name
                    AND tc.table_schema = kcu.table_schema
                JOIN t ON t.schema = tc.table_schema AND t.name = tc.table_name
                WHERE tc.constraint_type = 'PRIMARY KEY'
                ORDER BY tc.table_schema, tc.table_name, kcu.ordinal_position;
            """
            return await conn.fetch(pk_query, *params)

    async def get_indexes_info(self, connection, tables: list) -> list:
        """Get indexes information for PostgreSQL."""
        if not tables:
            return []

        pool = await connection.get_pool()
        async with pool.acquire() as conn:
            values_clause, params = self._build_table_filter_clause(tables)

            idx_query = f"""
                WITH t_filter(schema, name) AS (VALUES {values_clause})
                SELECT
                    ns.nspname      AS table_schema,
                    tcls.relname    AS table_name,
                    icls.relname    AS index_name,
                    ix.indisunique  AS is_unique,
                    am.amname       AS index_type,
                    string_agg(a.attname, ',' ORDER BY att.ordinality) AS column_names
                FROM pg_class tcls
                JOIN pg_namespace ns ON tcls.relnamespace = ns.oid
                JOIN pg_index ix ON tcls.oid = ix.indrelid
                JOIN pg_class icls ON icls.oid = ix.indexrelid
                JOIN pg_am am ON icls.relam = am.oid
                JOIN pg_attribute a ON a.attrelid = tcls.oid
                JOIN unnest(ix.indkey) WITH ORDINALITY AS att(attnum, ordinality) ON a.attnum = att.attnum
                JOIN t_filter ON t_filter.schema = ns.nspname AND t_filter.name = tcls.relname
                WHERE tcls.relkind = 'r'
                    AND icls.relname NOT LIKE '%_pkey'
                GROUP BY ns.nspname, tcls.relname, icls.relname, ix.indisunique, am.amname
                ORDER BY ns.nspname, tcls.relname, icls.relname;
            """
            return await conn.fetch(idx_query, *params)

    async def list_tables_info(self, connection) -> list[dict[str, Any]]:
        """Get list of tables with basic information for PostgreSQL."""
        pool = await connection.get_pool()
        async with pool.acquire() as conn:
            # Get table names and basic info without row counts for better performance
            tables_query = """
                SELECT
                    table_schema,
                    table_name,
                    table_type
                FROM information_schema.tables
                WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
                ORDER BY table_schema, table_name;
            """
            tables = await conn.fetch(tables_query)

            # Convert to expected format
            return [
                {
                    "table_schema": table["table_schema"],
                    "table_name": table["table_name"],
                    "table_type": table["table_type"],
                }
                for table in tables
            ]
