"""Database schema introspection utilities."""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, TypedDict

import aiosqlite
import duckdb

from sqlsaber.database.connection import (
    BaseDatabaseConnection,
    CSVConnection,
    DuckDBConnection,
    MySQLConnection,
    PostgreSQLConnection,
    SQLiteConnection,
)


class ColumnInfo(TypedDict):
    """Type definition for column information."""

    data_type: str
    nullable: bool
    default: str | None
    max_length: int | None
    precision: int | None
    scale: int | None


class ForeignKeyInfo(TypedDict):
    """Type definition for foreign key information."""

    column: str
    references: dict[str, str]  # {"table": "schema.table", "column": "column_name"}


class IndexInfo(TypedDict):
    """Type definition for index information."""

    name: str
    columns: list[str]  # ordered
    unique: bool
    type: str | None  # btree, gin, FULLTEXT, etc. None if unknown


class SchemaInfo(TypedDict):
    """Type definition for schema information."""

    schema: str
    name: str
    type: str
    columns: dict[str, ColumnInfo]
    primary_keys: list[str]
    foreign_keys: list[ForeignKeyInfo]
    indexes: list[IndexInfo]


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
    async def get_indexes_info(self, connection, tables: list) -> list:
        """Get indexes information for the specific database type."""
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

    async def get_indexes_info(self, connection, tables: list) -> list:
        """Get indexes information for PostgreSQL."""
        if not tables:
            return []

        pool = await connection.get_pool()
        async with pool.acquire() as conn:
            # Build proper table filters
            idx_table_filters = []
            for table in tables:
                idx_table_filters.append(
                    f"(ns.nspname = '{table['table_schema']}' AND t.relname = '{table['table_name']}')"
                )

            idx_query = f"""
                SELECT
                    ns.nspname      AS table_schema,
                    t.relname       AS table_name,
                    i.relname       AS index_name,
                    ix.indisunique  AS is_unique,
                    am.amname       AS index_type,
                    array_agg(a.attname ORDER BY ord.ordinality) AS column_names
                FROM pg_class t
                JOIN pg_namespace ns     ON ns.oid = t.relnamespace
                JOIN pg_index    ix      ON ix.indrelid = t.oid
                JOIN pg_class    i       ON i.oid  = ix.indexrelid
                JOIN pg_am       am      ON am.oid = i.relam
                JOIN LATERAL unnest(ix.indkey) WITH ORDINALITY AS ord(attnum, ordinality)
                     ON TRUE
                JOIN pg_attribute a      ON a.attrelid = t.oid AND a.attnum = ord.attnum
                WHERE ns.nspname NOT IN ('pg_catalog', 'information_schema')
                  AND ({" OR ".join(idx_table_filters)})
                GROUP BY table_schema, table_name, index_name, is_unique, index_type
                ORDER BY table_schema, table_name, index_name;
            """
            return await conn.fetch(idx_query)

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

    async def get_indexes_info(self, connection, tables: list) -> list:
        """Get indexes information for MySQL."""
        if not tables:
            return []

        pool = await connection.get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                # Build proper table filters
                idx_table_filters = []
                for table in tables:
                    idx_table_filters.append(
                        f"(TABLE_SCHEMA = '{table['table_schema']}' AND TABLE_NAME = '{table['table_name']}')"
                    )

                idx_query = f"""
                    SELECT
                        TABLE_SCHEMA   AS table_schema,
                        TABLE_NAME     AS table_name,
                        INDEX_NAME     AS index_name,
                        (NON_UNIQUE = 0) AS is_unique,
                        INDEX_TYPE     AS index_type,
                        GROUP_CONCAT(COLUMN_NAME ORDER BY SEQ_IN_INDEX) AS column_names
                    FROM INFORMATION_SCHEMA.STATISTICS
                    WHERE ({" OR ".join(idx_table_filters)})
                    GROUP BY table_schema, table_name, index_name, is_unique, index_type
                    ORDER BY table_schema, table_name, index_name;
                """
                await cursor.execute(idx_query)
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

    async def get_indexes_info(self, connection, tables: list) -> list:
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


class DuckDBSchemaIntrospector(BaseSchemaIntrospector):
    """DuckDB-specific schema introspection."""

    async def _execute_query(
        self,
        connection: DuckDBConnection | CSVConnection,
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

        if isinstance(connection, CSVConnection):
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
        where_conditions = [
            "table_schema NOT IN ('information_schema', 'pg_catalog', 'duckdb_catalog')"
        ]
        params: list[Any] = []

        if table_pattern:
            if "." in table_pattern:
                schema_pattern, table_name_pattern = table_pattern.split(".", 1)
                where_conditions.append(
                    "(table_schema LIKE ? AND table_name LIKE ?)"
                )
                params.extend([schema_pattern, table_name_pattern])
            else:
                where_conditions.append(
                    "(table_name LIKE ? OR table_schema || '.' || table_name LIKE ?)"
                )
                params.extend([table_pattern, table_pattern])

        query = f"""
            SELECT
                table_schema,
                table_name,
                table_type
            FROM information_schema.tables
            WHERE {" AND ".join(where_conditions)}
            ORDER BY table_schema, table_name;
        """

        return await self._execute_query(connection, query, tuple(params))

    async def get_columns_info(self, connection, tables: list) -> list[dict[str, Any]]:
        """Get columns information for DuckDB."""
        if not tables:
            return []

        table_filters = []
        for table in tables:
            table_filters.append(
                "(table_schema = ? AND table_name = ?)"
            )

        params: list[Any] = []
        for table in tables:
            params.extend([table["table_schema"], table["table_name"]])

        query = f"""
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
            WHERE {" OR ".join(table_filters)}
            ORDER BY table_schema, table_name, ordinal_position;
        """

        return await self._execute_query(connection, query, tuple(params))

    async def get_foreign_keys_info(self, connection, tables: list) -> list[dict[str, Any]]:
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

    async def get_primary_keys_info(self, connection, tables: list) -> list[dict[str, Any]]:
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

    async def get_indexes_info(self, connection, tables: list) -> list[dict[str, Any]]:
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
                    column_section = sql_text[sql_text.find("(") + 1 : sql_text.rfind(")")]
                    columns = [col.strip().strip('"') for col in column_section.split(",") if col.strip()]

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
        query = """
            SELECT
                table_schema,
                table_name,
                table_type
            FROM information_schema.tables
            WHERE table_schema NOT IN ('information_schema', 'pg_catalog', 'duckdb_catalog')
            ORDER BY table_schema, table_name;
        """

        return await self._execute_query(connection, query)


class SchemaManager:
    """Manages database schema introspection."""

    def __init__(self, db_connection: BaseDatabaseConnection):
        self.db = db_connection

        # Select appropriate introspector based on connection type
        if isinstance(db_connection, PostgreSQLConnection):
            self.introspector = PostgreSQLSchemaIntrospector()
        elif isinstance(db_connection, MySQLConnection):
            self.introspector = MySQLSchemaIntrospector()
        elif isinstance(db_connection, SQLiteConnection):
            self.introspector = SQLiteSchemaIntrospector()
        elif isinstance(db_connection, (DuckDBConnection, CSVConnection)):
            self.introspector = DuckDBSchemaIntrospector()
        else:
            raise ValueError(
                f"Unsupported database connection type: {type(db_connection)}"
            )

    async def get_schema_info(
        self, table_pattern: str | None = None
    ) -> dict[str, SchemaInfo]:
        """Get database schema information, optionally filtered by table pattern.

        Args:
            table_pattern: Optional SQL LIKE pattern to filter tables (e.g., 'public.user%')
        """
        # Get all schema components
        tables = await self.introspector.get_tables_info(self.db, table_pattern)
        columns = await self.introspector.get_columns_info(self.db, tables)
        foreign_keys = await self.introspector.get_foreign_keys_info(self.db, tables)
        primary_keys = await self.introspector.get_primary_keys_info(self.db, tables)
        indexes = await self.introspector.get_indexes_info(self.db, tables)

        # Build schema structure
        schema_info = self._build_table_structure(tables)
        self._add_columns_to_schema(schema_info, columns)
        self._add_primary_keys_to_schema(schema_info, primary_keys)
        self._add_foreign_keys_to_schema(schema_info, foreign_keys)
        self._add_indexes_to_schema(schema_info, indexes)

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
                "indexes": [],
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

    def _add_indexes_to_schema(
        self, schema_info: dict[str, dict], indexes: list
    ) -> None:
        """Add index information to schema."""
        for idx in indexes:
            full_name = f"{idx['table_schema']}.{idx['table_name']}"
            if full_name in schema_info:
                # Handle different column name formats from different databases
                if isinstance(idx["column_names"], list):
                    columns = idx["column_names"]
                else:
                    # MySQL returns comma-separated string
                    columns = (
                        idx["column_names"].split(",") if idx["column_names"] else []
                    )

                schema_info[full_name]["indexes"].append(
                    {
                        "name": idx["index_name"],
                        "columns": columns,
                        "unique": idx["is_unique"],
                        "type": idx.get("index_type"),
                    }
                )

    async def list_tables(self) -> dict[str, Any]:
        """Get a list of all tables with basic information."""
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

        return result
