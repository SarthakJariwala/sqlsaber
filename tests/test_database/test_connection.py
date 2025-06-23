"""Tests for database connection module."""

import pytest

from sqlsaber.database.connection import (
    DatabaseConnection,
    MySQLConnection,
    PostgreSQLConnection,
    SQLiteConnection,
)


class TestDatabaseConnectionFactory:
    """Test the DatabaseConnection factory function."""

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

    def test_unsupported_database(self):
        """Test error for unsupported database type."""
        with pytest.raises(ValueError, match="Unsupported database type"):
            DatabaseConnection("mongodb://localhost:27017/db")
