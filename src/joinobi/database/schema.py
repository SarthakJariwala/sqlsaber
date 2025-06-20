"""Database schema introspection utilities."""

import time
from typing import Any, Dict, Optional, Tuple

from joinobi.database.connection import DatabaseConnection
from joinobi.models.types import SchemaInfo


class SchemaManager:
    """Manages database schema introspection with caching."""

    def __init__(self, db_connection: DatabaseConnection, cache_ttl: int = 900):
        self.db = db_connection
        self.cache_ttl = cache_ttl  # Default 15 minutes
        self._schema_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}

    def clear_schema_cache(self):
        """Clear the schema cache."""
        self._schema_cache.clear()

    async def get_schema_info(
        self, table_pattern: Optional[str] = None
    ) -> Dict[str, SchemaInfo]:
        """Get database schema information, optionally filtered by table pattern.

        Args:
            table_pattern: Optional SQL LIKE pattern to filter tables (e.g., 'public.user%')
        """
        # Create cache key
        cache_key = f"schema:{table_pattern or 'all'}"

        # Check cache first
        if cache_key in self._schema_cache:
            cached_time, cached_data = self._schema_cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                return cached_data

        # If not in cache or expired, fetch from database
        pool = await self.db.get_pool()
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
            tables = await conn.fetch(tables_query, *params)

            # Get columns for filtered tables only
            if tables:
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
                columns = await conn.fetch(columns_query)
            else:
                columns = []

            # Get foreign keys for filtered tables
            if tables:
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
                        AND ({" OR ".join(table_filters)});
                """
                foreign_keys = await conn.fetch(fk_query)
            else:
                foreign_keys = []

            # Get primary keys for filtered tables
            if tables:
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
                        AND ({" OR ".join(table_filters)})
                    ORDER BY tc.table_schema, tc.table_name, kcu.ordinal_position;
                """
                primary_keys = await conn.fetch(pk_query)
            else:
                primary_keys = []

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

            # Cache the result
            self._schema_cache[cache_key] = (time.time(), schema_info)

            return schema_info

    async def list_tables(self) -> Dict[str, Any]:
        """Get a list of all tables with basic information like row counts."""
        pool = await self.db.get_pool()
        async with pool.acquire() as conn:
            # Get tables with row counts
            tables_query = """
                WITH table_stats AS (
                    SELECT
                        schemaname,
                        tablename,
                        n_live_tup as approximate_row_count
                    FROM pg_stat_user_tables
                )
                SELECT
                    t.table_schema,
                    t.table_name,
                    t.table_type,
                    COALESCE(ts.approximate_row_count, 0) as row_count
                FROM information_schema.tables t
                LEFT JOIN table_stats ts 
                    ON t.table_schema = ts.schemaname 
                    AND t.table_name = ts.tablename
                WHERE t.table_schema NOT IN ('pg_catalog', 'information_schema')
                ORDER BY t.table_schema, t.table_name;
            """

            tables = await conn.fetch(tables_query)

            # Format the result
            result = {"tables": [], "total_tables": len(tables)}

            for table in tables:
                result["tables"].append(
                    {
                        "schema": table["table_schema"],
                        "name": table["table_name"],
                        "full_name": f"{table['table_schema']}.{table['table_name']}",
                        "type": table["table_type"],
                        "row_count": table["row_count"],
                    }
                )

            return result
