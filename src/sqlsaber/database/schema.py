"""Database schema management."""

from typing import Any

from .base import (
    BaseDatabaseConnection,
    ColumnInfo,
    ForeignKeyInfo,
    IndexInfo,
    SchemaInfo,
)
from .csv import CSVConnection
from .duckdb import DuckDBConnection, DuckDBSchemaIntrospector
from .mysql import MySQLConnection, MySQLSchemaIntrospector
from .postgresql import PostgreSQLConnection, PostgreSQLSchemaIntrospector
from .sqlite import SQLiteConnection, SQLiteSchemaIntrospector


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

    def _add_columns_to_schema(self, schema_info: dict, columns: list) -> None:
        """Add column information to schema structure."""
        for col in columns:
            full_name = f"{col['table_schema']}.{col['table_name']}"
            if full_name in schema_info:
                column_info: ColumnInfo = {
                    "data_type": col["data_type"],
                    "nullable": col.get("is_nullable", "YES") == "YES",
                    "default": col.get("column_default"),
                    "max_length": col.get("character_maximum_length"),
                    "precision": col.get("numeric_precision"),
                    "scale": col.get("numeric_scale"),
                }
                # Add type field for display compatibility
                column_info["type"] = col["data_type"]
                schema_info[full_name]["columns"][col["column_name"]] = column_info

    def _add_primary_keys_to_schema(
        self, schema_info: dict, primary_keys: list
    ) -> None:
        """Add primary key information to schema structure."""
        for pk in primary_keys:
            full_name = f"{pk['table_schema']}.{pk['table_name']}"
            if full_name in schema_info:
                schema_info[full_name]["primary_keys"].append(pk["column_name"])

    def _add_foreign_keys_to_schema(
        self, schema_info: dict, foreign_keys: list
    ) -> None:
        """Add foreign key information to schema structure."""
        for fk in foreign_keys:
            full_name = f"{fk['table_schema']}.{fk['table_name']}"
            if full_name in schema_info:
                fk_info: ForeignKeyInfo = {
                    "column": fk["column_name"],
                    "references": {
                        "table": f"{fk['foreign_table_schema']}.{fk['foreign_table_name']}",
                        "column": fk["foreign_column_name"],
                    },
                }
                schema_info[full_name]["foreign_keys"].append(fk_info)

    def _add_indexes_to_schema(self, schema_info: dict, indexes: list) -> None:
        """Add index information to schema structure."""
        for idx in indexes:
            full_name = f"{idx['table_schema']}.{idx['table_name']}"
            if full_name in schema_info:
                # Handle column names - could be comma-separated string or list
                if isinstance(idx.get("column_names"), str):
                    columns = [
                        col.strip()
                        for col in idx["column_names"].split(",")
                        if col.strip()
                    ]
                elif isinstance(idx.get("column_names"), list):
                    columns = idx["column_names"]
                else:
                    columns = []

                index_info: IndexInfo = {
                    "name": idx["index_name"],
                    "columns": columns,
                    "unique": bool(idx.get("is_unique", False)),
                    "type": idx.get("index_type"),
                }
                schema_info[full_name]["indexes"].append(index_info)

    async def list_tables(self) -> dict[str, Any]:
        """Get list of tables with basic information."""
        tables_list = await self.introspector.list_tables_info(self.db)

        # Add full_name and name fields for backwards compatibility
        for table in tables_list:
            table["full_name"] = f"{table['table_schema']}.{table['table_name']}"
            table["name"] = table["table_name"]
            table["schema"] = table["table_schema"]
            table["type"] = table["table_type"]  # Map table_type to type for display

        return {"tables": tables_list, "total_tables": len(tables_list)}

    async def close(self):
        """Close database connection."""
        await self.db.close()
