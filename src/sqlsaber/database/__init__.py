"""Database module for SQLSaber."""

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
from .duckdb import DuckDBConnection, DuckDBSchemaIntrospector
from .mysql import MySQLConnection, MySQLSchemaIntrospector
from .postgresql import PostgreSQLConnection, PostgreSQLSchemaIntrospector
from .schema import SchemaManager
from .sqlite import SQLiteConnection, SQLiteSchemaIntrospector


def DatabaseConnection(connection_string: str) -> BaseDatabaseConnection:
    """Factory function to create appropriate database connection based on connection string."""
    if connection_string.startswith("postgresql://"):
        return PostgreSQLConnection(connection_string)
    elif connection_string.startswith("mysql://"):
        return MySQLConnection(connection_string)
    elif connection_string.startswith("sqlite:///"):
        return SQLiteConnection(connection_string)
    elif connection_string.startswith("duckdb://"):
        return DuckDBConnection(connection_string)
    elif connection_string.startswith("csv:///"):
        return CSVConnection(connection_string)
    else:
        raise ValueError(
            f"Unsupported database type in connection string: {connection_string}"
        )


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
    "PostgreSQLSchemaIntrospector",
    "MySQLSchemaIntrospector",
    "SQLiteSchemaIntrospector",
    "DuckDBSchemaIntrospector",
    "CSVSchemaIntrospector",
    # Factory function and manager
    "DatabaseConnection",
    "SchemaManager",
]
