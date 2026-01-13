"""Database module for SQLSaber."""

from collections.abc import Iterable

from .base import (
    DEFAULT_QUERY_TIMEOUT,
    BaseDatabaseConnection,
    BaseSchemaIntrospector,
    ColumnInfo,
    ForeignKeyInfo,
    IndexInfo,
    QueryTimeoutError,
    SchemaInfo,
)
from .csv import CSVConnection, CSVSchemaIntrospector
from .csvs import CSVsConnection
from .duckdb import DuckDBConnection, DuckDBSchemaIntrospector
from .mysql import MySQLConnection, MySQLSchemaIntrospector
from .postgresql import PostgreSQLConnection, PostgreSQLSchemaIntrospector
from .schema import SchemaManager
from .sqlite import SQLiteConnection, SQLiteSchemaIntrospector


def DatabaseConnection(
    connection_string: str, *, excluded_schemas: Iterable[str] | None = None
) -> BaseDatabaseConnection:
    """Factory function to create appropriate database connection based on connection string."""
    if connection_string.startswith("postgresql://"):
        conn = PostgreSQLConnection(connection_string)
    elif connection_string.startswith("mysql://"):
        conn = MySQLConnection(connection_string)
    elif connection_string.startswith("sqlite:///"):
        conn = SQLiteConnection(connection_string)
    elif connection_string.startswith("duckdb://"):
        conn = DuckDBConnection(connection_string)
    elif connection_string.startswith("csv:///"):
        conn = CSVConnection(connection_string)
    elif connection_string.startswith("csvs://"):
        conn = CSVsConnection(connection_string)
    else:
        raise ValueError(
            f"Unsupported database type in connection string: {connection_string}"
        )

    conn.set_excluded_schemas(excluded_schemas)
    return conn


__all__ = [
    # Base classes and types
    "BaseDatabaseConnection",
    "BaseSchemaIntrospector",
    "ColumnInfo",
    "DEFAULT_QUERY_TIMEOUT",
    "ForeignKeyInfo",
    "IndexInfo",
    "QueryTimeoutError",
    "SchemaInfo",
    # Concrete implementations
    "PostgreSQLConnection",
    "MySQLConnection",
    "SQLiteConnection",
    "DuckDBConnection",
    "CSVConnection",
    "CSVsConnection",
    "PostgreSQLSchemaIntrospector",
    "MySQLSchemaIntrospector",
    "SQLiteSchemaIntrospector",
    "DuckDBSchemaIntrospector",
    "CSVSchemaIntrospector",
    # Factory function and manager
    "DatabaseConnection",
    "SchemaManager",
]
