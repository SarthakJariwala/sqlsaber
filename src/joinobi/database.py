import os
from typing import Any, Dict, Optional

import asyncpg


class DatabaseConnection:
    """Manages database connections for the SQL agent."""

    def __init__(self, connection_string: Optional[str] = None):
        self.connection_string = connection_string or os.getenv(
            "DATABASE_URL", "postgresql://localhost:5432/postgres"
        )
        self._pool: Optional[asyncpg.Pool] = None

    async def get_pool(self) -> asyncpg.Pool:
        """Get or create connection pool."""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                self.connection_string, min_size=1, max_size=10
            )
        return self._pool

    async def close(self):
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None

    async def execute_query(self, query: str, *args) -> list[Dict[str, Any]]:
        """Execute a query and return results as list of dicts."""
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *args)
            return [dict(row) for row in rows]

    async def get_schema_info(self) -> Dict[str, Any]:
        """Get database schema information."""
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            # Get tables
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

            # Get columns for each table
            columns_query = """
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
                WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
                ORDER BY table_schema, table_name, ordinal_position;
            """
            columns = await conn.fetch(columns_query)

            # Get foreign keys
            fk_query = """
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
                    AND tc.table_schema NOT IN ('pg_catalog', 'information_schema');
            """
            foreign_keys = await conn.fetch(fk_query)

            # Get primary keys
            pk_query = """
                SELECT
                    tc.table_schema,
                    tc.table_name,
                    kcu.column_name
                FROM information_schema.table_constraints AS tc
                JOIN information_schema.key_column_usage AS kcu
                    ON tc.constraint_name = kcu.constraint_name
                    AND tc.table_schema = kcu.table_schema
                WHERE tc.constraint_type = 'PRIMARY KEY'
                    AND tc.table_schema NOT IN ('pg_catalog', 'information_schema')
                ORDER BY tc.table_schema, tc.table_name, kcu.ordinal_position;
            """
            primary_keys = await conn.fetch(pk_query)

            # Organize the data
            schema_info = {}

            # Build table structure
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

            # Add columns
            for col in columns:
                full_name = f"{col['table_schema']}.{col['table_name']}"
                if full_name in schema_info:
                    col_info = {
                        "data_type": col["data_type"],
                        "nullable": col["is_nullable"] == "YES",
                        "default": col["column_default"],
                    }

                    if col["character_maximum_length"]:
                        col_info["max_length"] = col["character_maximum_length"]
                    if col["numeric_precision"]:
                        col_info["precision"] = col["numeric_precision"]
                    if col["numeric_scale"]:
                        col_info["scale"] = col["numeric_scale"]

                    schema_info[full_name]["columns"][col["column_name"]] = col_info

            # Add primary keys
            for pk in primary_keys:
                full_name = f"{pk['table_schema']}.{pk['table_name']}"
                if full_name in schema_info:
                    schema_info[full_name]["primary_keys"].append(pk["column_name"])

            # Add foreign keys
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

            return schema_info
