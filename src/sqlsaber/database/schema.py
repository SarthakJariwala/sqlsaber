"""Database schema introspection utilities."""

import time
from abc import ABC, abstractmethod
from typing import Any

import aiosqlite

from sqlsaber.database.connection import (
    BaseDatabaseConnection,
    CSVConnection,
    MySQLConnection,
    PostgreSQLConnection,
    SQLiteConnection,
)
from sqlsaber.models.types import SchemaInfo


class BaseSchemaIntrospector(ABC):
    """Abstract base class for database-specific schema introspection."""

    @abstractmethod
    async def get_tables_info(
        self, connection, table_pattern: str | None = None
    ) -> dict[str, Any]:
        """Get tables information for the specific database type."""
        pass

    @abstractmethod
    async def get_columns_info(self, connection, tables: list) -> list:
        """Get columns information for the specific database type."""
        pass

    @abstractmethod
    async def get_foreign_keys_info(self, connection, tables: list) -> list:
        """Get foreign keys information for the specific database type."""
        pass

    @abstractmethod
    async def get_primary_keys_info(self, connection, tables: list) -> list:
        """Get primary keys information for the specific database type."""
        pass

    @abstractmethod
    async def list_tables_info(self, connection) -> list[dict[str, Any]]:
        """Get list of tables with basic information."""
        pass


class PostgreSQLSchemaIntrospector(BaseSchemaIntrospector):
    """PostgreSQL-specific schema introspection."""

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
            # Build IN clause for the tables we found
            table_filters = []
            for table in tables:
                table_filters.append(
                    f"(table_schema = '{table['table_schema']}' AND table_name = '{table['table_name']}')"
                )

            columns_query = f"""
                SELECT
                    table_schema,
                    table_name,
                    column_name,
                    data_type,
                    is_nullable,
                    column_default,
                    character_maximum_length,
                    numeric_precision,
                    numeric_scale
                FROM information_schema.columns
                WHERE ({" OR ".join(table_filters)})
                ORDER BY table_schema, table_name, ordinal_position;
            """
            return await conn.fetch(columns_query)

    async def get_foreign_keys_info(self, connection, tables: list) -> list:
        """Get foreign keys information for PostgreSQL."""
        if not tables:
            return []

        pool = await connection.get_pool()
        async with pool.acquire() as conn:
            # Build proper table filters with tc. prefix
            fk_table_filters = []
            for table in tables:
                fk_table_filters.append(
                    f"(tc.table_schema = '{table['table_schema']}' AND tc.table_name = '{table['table_name']}')"
                )

            fk_query = f"""
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
                WHERE tc.constraint_type = 'FOREIGN KEY'
                    AND ({" OR ".join(fk_table_filters)});
            """
            return await conn.fetch(fk_query)

    async def get_primary_keys_info(self, connection, tables: list) -> list:
        """Get primary keys information for PostgreSQL."""
        if not tables:
            return []

        pool = await connection.get_pool()
        async with pool.acquire() as conn:
            # Build proper table filters with tc. prefix
            pk_table_filters = []
            for table in tables:
                pk_table_filters.append(
                    f"(tc.table_schema = '{table['table_schema']}' AND tc.table_name = '{table['table_name']}')"
                )

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
                    AND ({" OR ".join(pk_table_filters)})
                ORDER BY tc.table_schema, tc.table_name, kcu.ordinal_position;
            """
            return await conn.fetch(pk_query)

    async def list_tables_info(self, connection) -> list[dict[str, Any]]:
        """Get list of tables with basic information for PostgreSQL."""
        pool = await connection.get_pool()
        async with pool.acquire() as conn:
            # Get tables without row counts for better performance
            tables_query = """
                SELECT
                    t.table_schema,
                    t.table_name,
                    t.table_type
                FROM information_schema.tables t
                WHERE t.table_schema NOT IN ('pg_catalog', 'information_schema')
                ORDER BY t.table_schema, t.table_name;
            """
            records = await conn.fetch(tables_query)

            # Convert asyncpg.Record objects to dictionaries
            return [
                {
                    "table_schema": record["table_schema"],
                    "table_name": record["table_name"],
                    "table_type": record["table_type"],
                }
                for record in records
            ]


class MySQLSchemaIntrospector(BaseSchemaIntrospector):
    """MySQL-specific schema introspection."""

    async def get_tables_info(
        self, connection, table_pattern: str | None = None
    ) -> dict[str, Any]:
        """Get tables information for MySQL."""
        pool = await connection.get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                # Build WHERE clause for filtering
                where_conditions = [
                    "table_schema NOT IN ('information_schema', 'performance_schema', 'mysql', 'sys')"
                ]
                params = []

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
                await cursor.execute(tables_query, params)
                return await cursor.fetchall()

    async def get_columns_info(self, connection, tables: list) -> list:
        """Get columns information for MySQL."""
        if not tables:
            return []

        pool = await connection.get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                # Build IN clause for the tables we found
                table_filters = []
                for table in tables:
                    table_filters.append(
                        f"(table_schema = '{table['table_schema']}' AND table_name = '{table['table_name']}')"
                    )

                columns_query = f"""
                    SELECT
                        table_schema,
                        table_name,
                        column_name,
                        data_type,
                        is_nullable,
                        column_default,
                        character_maximum_length,
                        numeric_precision,
                        numeric_scale
                    FROM information_schema.columns
                    WHERE ({" OR ".join(table_filters)})
                    ORDER BY table_schema, table_name, ordinal_position;
                """
                await cursor.execute(columns_query)
                return await cursor.fetchall()

    async def get_foreign_keys_info(self, connection, tables: list) -> list:
        """Get foreign keys information for MySQL."""
        if not tables:
            return []

        pool = await connection.get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                # Build proper table filters
                fk_table_filters = []
                for table in tables:
                    fk_table_filters.append(
                        f"(tc.table_schema = '{table['table_schema']}' AND tc.table_name = '{table['table_name']}')"
                    )

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
                        AND ({" OR ".join(fk_table_filters)});
                """
                await cursor.execute(fk_query)
                return await cursor.fetchall()

    async def get_primary_keys_info(self, connection, tables: list) -> list:
        """Get primary keys information for MySQL."""
        if not tables:
            return []

        pool = await connection.get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                # Build proper table filters
                pk_table_filters = []
                for table in tables:
                    pk_table_filters.append(
                        f"(tc.table_schema = '{table['table_schema']}' AND tc.table_name = '{table['table_name']}')"
                    )

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
                        AND ({" OR ".join(pk_table_filters)})
                    ORDER BY tc.table_schema, tc.table_name, kcu.ordinal_position;
                """
                await cursor.execute(pk_query)
                return await cursor.fetchall()

    async def list_tables_info(self, connection) -> list[dict[str, Any]]:
        """Get list of tables with basic information for MySQL."""
        pool = await connection.get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                # Get tables without row counts for better performance
                tables_query = """
                    SELECT
                        t.table_schema,
                        t.table_name,
                        t.table_type
                    FROM information_schema.tables t
                    WHERE t.table_schema NOT IN ('information_schema', 'performance_schema', 'mysql', 'sys')
                    ORDER BY t.table_schema, t.table_name;
                """
                await cursor.execute(tables_query)
                rows = await cursor.fetchall()

                # Convert rows to dictionaries
                return [
                    {
                        "table_schema": row["table_schema"],
                        "table_name": row["table_name"],
                        "table_type": row["table_type"],
                    }
                    for row in rows
                ]


class SQLiteSchemaIntrospector(BaseSchemaIntrospector):
    """SQLite-specific schema introspection."""

    async def _execute_query(self, connection, query: str, params=()) -> list:
        """Helper method to execute queries on both SQLite and CSV connections."""
        # Handle both SQLite and CSV connections
        if hasattr(connection, "database_path"):
            # Regular SQLite connection
            async with aiosqlite.connect(connection.database_path) as conn:
                conn.row_factory = aiosqlite.Row
                cursor = await conn.execute(query, params)
                return await cursor.fetchall()
        else:
            # CSV connection - use the existing connection
            conn = await connection.get_pool()
            cursor = await conn.execute(query, params)
            return await cursor.fetchall()

    async def get_tables_info(
        self, connection, table_pattern: str | None = None
    ) -> dict[str, Any]:
        """Get tables information for SQLite."""
        where_conditions = ["type IN ('table', 'view')", "name NOT LIKE 'sqlite_%'"]
        params = ()

        if table_pattern:
            where_conditions.append("name LIKE ?")
            params = (table_pattern,)

        query = f"""
            SELECT
                'main' as table_schema,
                name as table_name,
                type as table_type
            FROM sqlite_master
            WHERE {" AND ".join(where_conditions)}
            ORDER BY name;
        """

        return await self._execute_query(connection, query, params)

    async def get_columns_info(self, connection, tables: list) -> list:
        """Get columns information for SQLite."""
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
                    }
                )

        return columns

    async def get_foreign_keys_info(self, connection, tables: list) -> list:
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

    async def get_primary_keys_info(self, connection, tables: list) -> list:
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

    async def list_tables_info(self, connection) -> list[dict[str, Any]]:
        """Get list of tables with basic information for SQLite."""
        # Get table names without row counts for better performance
        tables_query = """
            SELECT
                'main' as table_schema,
                name as table_name,
                type as table_type
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
            }
            for table in tables
        ]


class SchemaManager:
    """Manages database schema introspection with caching."""

    def __init__(self, db_connection: BaseDatabaseConnection, cache_ttl: int = 900):
        self.db = db_connection
        self.cache_ttl = cache_ttl  # Default 15 minutes
        self._schema_cache: dict[str, tuple[float, dict[str, Any]]] = {}

        # Select appropriate introspector based on connection type
        if isinstance(db_connection, PostgreSQLConnection):
            self.introspector = PostgreSQLSchemaIntrospector()
        elif isinstance(db_connection, MySQLConnection):
            self.introspector = MySQLSchemaIntrospector()
        elif isinstance(db_connection, (SQLiteConnection, CSVConnection)):
            self.introspector = SQLiteSchemaIntrospector()
        else:
            raise ValueError(
                f"Unsupported database connection type: {type(db_connection)}"
            )

    def clear_schema_cache(self):
        """Clear the schema cache."""
        self._schema_cache.clear()

    async def get_schema_info(
        self, table_pattern: str | None = None
    ) -> dict[str, SchemaInfo]:
        """Get database schema information, optionally filtered by table pattern.

        Args:
            table_pattern: Optional SQL LIKE pattern to filter tables (e.g., 'public.user%')
        """
        # Check cache first
        cache_key = f"schema:{table_pattern or 'all'}"
        cached_data = self._get_cached_schema(cache_key)
        if cached_data is not None:
            return cached_data

        # Fetch from database if not cached
        schema_info = await self._fetch_schema_from_db(table_pattern)

        # Cache the result
        self._schema_cache[cache_key] = (time.time(), schema_info)
        return schema_info

    def _get_cached_schema(self, cache_key: str) -> dict[str, SchemaInfo] | None:
        """Get schema from cache if available and not expired."""
        if cache_key in self._schema_cache:
            cached_time, cached_data = self._schema_cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                return cached_data
        return None

    async def _fetch_schema_from_db(
        self, table_pattern: str | None
    ) -> dict[str, SchemaInfo]:
        """Fetch schema information from database."""
        # Get all schema components
        tables = await self.introspector.get_tables_info(self.db, table_pattern)
        columns = await self.introspector.get_columns_info(self.db, tables)
        foreign_keys = await self.introspector.get_foreign_keys_info(self.db, tables)
        primary_keys = await self.introspector.get_primary_keys_info(self.db, tables)

        # Build schema structure
        schema_info = self._build_table_structure(tables)
        self._add_columns_to_schema(schema_info, columns)
        self._add_primary_keys_to_schema(schema_info, primary_keys)
        self._add_foreign_keys_to_schema(schema_info, foreign_keys)

        return schema_info

    def _build_table_structure(self, tables: list) -> dict[str, dict]:
        """Build basic table structure from table info."""
        schema_info = {}
        for table in tables:
            schema_name = table["table_schema"]
            table_name = table["table_name"]
            full_name = f"{schema_name}.{table_name}"

            schema_info[full_name] = {
                "schema": schema_name,
                "name": table_name,
                "type": table["table_type"],
                "columns": {},
                "primary_keys": [],
                "foreign_keys": [],
            }
        return schema_info

    def _add_columns_to_schema(
        self, schema_info: dict[str, dict], columns: list
    ) -> None:
        """Add column information to schema."""
        for col in columns:
            full_name = f"{col['table_schema']}.{col['table_name']}"
            if full_name in schema_info:
                col_info = {
                    "data_type": col["data_type"],
                    "nullable": col["is_nullable"] == "YES",
                    "default": col["column_default"],
                }

                # Add optional attributes
                for attr_map in [
                    ("character_maximum_length", "max_length"),
                    ("numeric_precision", "precision"),
                    ("numeric_scale", "scale"),
                ]:
                    if col.get(attr_map[0]):
                        col_info[attr_map[1]] = col[attr_map[0]]

                schema_info[full_name]["columns"][col["column_name"]] = col_info

    def _add_primary_keys_to_schema(
        self, schema_info: dict[str, dict], primary_keys: list
    ) -> None:
        """Add primary key information to schema."""
        for pk in primary_keys:
            full_name = f"{pk['table_schema']}.{pk['table_name']}"
            if full_name in schema_info:
                schema_info[full_name]["primary_keys"].append(pk["column_name"])

    def _add_foreign_keys_to_schema(
        self, schema_info: dict[str, dict], foreign_keys: list
    ) -> None:
        """Add foreign key information to schema."""
        for fk in foreign_keys:
            full_name = f"{fk['table_schema']}.{fk['table_name']}"
            if full_name in schema_info:
                schema_info[full_name]["foreign_keys"].append(
                    {
                        "column": fk["column_name"],
                        "references": {
                            "table": f"{fk['foreign_table_schema']}.{fk['foreign_table_name']}",
                            "column": fk["foreign_column_name"],
                        },
                    }
                )

    async def list_tables(self) -> dict[str, Any]:
        """Get a list of all tables with basic information."""
        # Check cache first
        cache_key = "list_tables"
        cached_data = self._get_cached_tables(cache_key)
        if cached_data is not None:
            return cached_data

        # Fetch from database if not cached
        tables = await self.introspector.list_tables_info(self.db)

        # Format the result
        result = {"tables": [], "total_tables": len(tables)}

        for table in tables:
            result["tables"].append(
                {
                    "schema": table["table_schema"],
                    "name": table["table_name"],
                    "full_name": f"{table['table_schema']}.{table['table_name']}",
                    "type": table["table_type"],
                }
            )

        # Cache the result
        self._schema_cache[cache_key] = (time.time(), result)
        return result

    def _get_cached_tables(self, cache_key: str) -> dict[str, Any] | None:
        """Get table list from cache if available and not expired."""
        if cache_key in self._schema_cache:
            cached_time, cached_data = self._schema_cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                return cached_data
        return None
