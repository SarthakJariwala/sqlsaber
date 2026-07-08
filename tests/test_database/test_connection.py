"""Tests for database connection module."""

from urllib.parse import urlencode

import pytest

from sqlsaber.database import (
    DatabaseConnection,
    DuckDBConnection,
    MySQLConnection,
    PostgreSQLConnection,
    SQLiteConnection,
)
from sqlsaber.database.csv import CSVConnection
from sqlsaber.database.csvs import CSVsConnection


class TestDatabaseConnectionFactory:
    """Test the DatabaseConnection factory function."""

    def test_csv_compatibility_exports_are_removed(self):
        """CSV implementation classes should be imported from their modules."""
        import sqlsaber.database as database
        import sqlsaber.database.csv as csv_module

        assert not hasattr(database, "CSVConnection")
        assert not hasattr(database, "CSVsConnection")
        assert not hasattr(database, "CSVSchemaIntrospector")
        assert not hasattr(csv_module, "CSVSchemaIntrospector")

    def test_postgresql_connection(self):
        """Test creating a PostgreSQL connection."""
        conn_string = "postgresql://user:pass@localhost:5432/db"
        conn = DatabaseConnection(conn_string)
        assert isinstance(conn, PostgreSQLConnection)
        assert conn.connection_string == conn_string

    def test_mysql_connection(self):
        """Test creating a MySQL connection."""
        conn_string = "mysql://user:pass@localhost:3306/db"
        conn = DatabaseConnection(conn_string)
        assert isinstance(conn, MySQLConnection)
        assert conn.connection_string == conn_string

    def test_sqlite_connection(self):
        """Test creating a SQLite connection."""
        conn_string = "sqlite:///path/to/db.sqlite"
        conn = DatabaseConnection(conn_string)
        assert isinstance(conn, SQLiteConnection)
        assert conn.connection_string == conn_string
        assert conn.database_path == "path/to/db.sqlite"

    def test_duckdb_connection(self):
        """Test creating a DuckDB connection."""
        conn_string = "duckdb:///path/to/data.duckdb"
        conn = DatabaseConnection(conn_string)
        assert isinstance(conn, DuckDBConnection)
        assert conn.connection_string == conn_string
        assert conn.database_path == "path/to/data.duckdb"

    def test_csv_connection(self):
        """Test creating a CSV connection."""
        conn_string = "csv:///path/to/data.csv"
        conn = DatabaseConnection(conn_string)
        assert isinstance(conn, CSVConnection)
        assert conn.connection_string == conn_string
        assert conn.csv_path == "path/to/data.csv"

    def test_csvs_connection(self):
        """Test creating a multi-CSV connection."""
        specs = ["csv:///path/to/users.csv", "csv:///path/to/orders.csv"]
        conn_string = f"csvs:///?{urlencode({'spec': specs}, doseq=True)}"
        conn = DatabaseConnection(conn_string)
        assert isinstance(conn, CSVsConnection)
        assert conn.connection_string == conn_string
        assert [source.csv_path for source in conn.csv_sources] == [
            "path/to/users.csv",
            "path/to/orders.csv",
        ]

    def test_unsupported_database(self):
        """Test error for unsupported database type."""
        with pytest.raises(ValueError, match="Unsupported database type"):
            DatabaseConnection("mongodb://localhost:27017/db")
