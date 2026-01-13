"""Tests for database resolver functionality."""

from unittest.mock import Mock, patch

import pytest

from sqlsaber.config.database import DatabaseConfig
from sqlsaber.database.resolver import DatabaseResolutionError, resolve_database


class TestDatabaseResolver:
    """Test cases for database resolution logic."""

    def test_resolve_connection_strings(self):
        """Test that connection strings are handled correctly."""
        config_mgr = Mock()

        # PostgreSQL connection string
        result = resolve_database("postgresql://user:pass@host:5432/testdb", config_mgr)
        assert result.name == "testdb"
        assert result.connection_string == "postgresql://user:pass@host:5432/testdb"
        assert result.excluded_schemas == []

        # MySQL connection string
        result = resolve_database("mysql://user:pass@host:3306/mydb", config_mgr)
        assert result.name == "mydb"
        assert result.connection_string == "mysql://user:pass@host:3306/mydb"
        assert result.excluded_schemas == []

        # SQLite connection string
        result = resolve_database("sqlite:///test.db", config_mgr)
        assert result.name == "test"
        assert result.connection_string == "sqlite:///test.db"
        assert result.excluded_schemas == []

        # CSV connection string
        result = resolve_database("csv:///data.csv", config_mgr)
        assert result.name == "data"
        assert result.connection_string == "csv:///data.csv"
        assert result.excluded_schemas == []

        # DuckDB connection string
        result = resolve_database("duckdb:///path/to/data.duckdb", config_mgr)
        assert result.name == "data"
        assert result.connection_string == "duckdb:///path/to/data.duckdb"
        assert result.excluded_schemas == []

    @patch("pathlib.Path.exists")
    def test_resolve_file_paths(self, mock_exists):
        """Test that file paths are resolved correctly."""
        mock_exists.return_value = True
        config_mgr = Mock()

        # CSV file
        result = resolve_database("data.csv", config_mgr)
        assert result.name == "data"
        assert result.connection_string.startswith("csv:///")
        assert result.connection_string.endswith("data.csv")
        assert result.excluded_schemas == []

        # SQLite file
        result = resolve_database("test.db", config_mgr)
        assert result.name == "test"
        assert result.connection_string.startswith("sqlite:///")
        assert result.connection_string.endswith("test.db")
        assert result.excluded_schemas == []

        # DuckDB file
        result = resolve_database("data.duckdb", config_mgr)
        assert result.name == "data"
        assert result.connection_string.startswith("duckdb:///")
        assert result.connection_string.endswith("data.duckdb")
        assert result.excluded_schemas == []

    @patch("pathlib.Path.exists")
    def test_resolve_multiple_csv_file_paths(self, mock_exists):
        """Test that multiple CSV paths resolve to a combined DuckDB view connection."""
        mock_exists.return_value = True
        config_mgr = Mock()

        result = resolve_database(["a.csv", "b.csv"], config_mgr)
        assert result.connection_string.startswith("csvs:///?")
        assert "spec=" in result.connection_string
        assert result.excluded_schemas == []

    def test_multiple_database_args_must_be_csv(self):
        config_mgr = Mock()

        with pytest.raises(
            DatabaseResolutionError,
            match="Multiple database arguments are only supported",
        ):
            resolve_database(["csv:///a.csv", "sqlite:///test.db"], config_mgr)

    @patch("pathlib.Path.exists")
    def test_file_not_found_error(self, mock_exists):
        """Test that missing files raise appropriate errors."""
        mock_exists.return_value = False
        config_mgr = Mock()

        with pytest.raises(
            DatabaseResolutionError, match="CSV file 'missing.csv' not found"
        ):
            resolve_database("missing.csv", config_mgr)

        with pytest.raises(
            DatabaseResolutionError, match="SQLite file 'missing.db' not found"
        ):
            resolve_database("missing.db", config_mgr)

        with pytest.raises(
            DatabaseResolutionError, match="DuckDB file 'missing.duckdb' not found"
        ):
            resolve_database("missing.duckdb", config_mgr)

    def test_resolve_configured_database(self):
        """Test that configured database names are resolved."""
        config_mgr = Mock()
        db_config = Mock(spec=DatabaseConfig)
        db_config.name = "mydb"
        db_config.to_connection_string.return_value = "postgresql://localhost:5432/mydb"
        db_config.exclude_schemas = ["foo"]
        config_mgr.get_database.return_value = db_config

        result = resolve_database("mydb", config_mgr)
        assert result.name == "mydb"
        assert result.connection_string == "postgresql://localhost:5432/mydb"
        assert result.excluded_schemas == ["foo"]

    def test_configured_database_not_found(self):
        """Test error when configured database doesn't exist."""
        config_mgr = Mock()
        config_mgr.get_database.return_value = None

        with pytest.raises(
            DatabaseResolutionError, match="Database connection 'unknown' not found"
        ):
            resolve_database("unknown", config_mgr)

    def test_resolve_default_database(self):
        """Test that None resolves to default database."""
        config_mgr = Mock()
        db_config = Mock(spec=DatabaseConfig)
        db_config.name = "default"
        db_config.to_connection_string.return_value = (
            "postgresql://localhost:5432/default"
        )
        db_config.exclude_schemas = ["bar"]
        config_mgr.get_default_database.return_value = db_config

        result = resolve_database(None, config_mgr)
        assert result.name == "default"
        assert result.connection_string == "postgresql://localhost:5432/default"
        assert result.excluded_schemas == ["bar"]

    def test_no_default_database_error(self):
        """Test error when no default database is configured."""
        config_mgr = Mock()
        config_mgr.get_default_database.return_value = None

        with pytest.raises(
            DatabaseResolutionError, match="No database connections configured"
        ):
            resolve_database(None, config_mgr)

    def test_connection_string_edge_cases(self):
        """Test edge cases in connection string parsing."""
        config_mgr = Mock()

        # PostgreSQL without database name
        result = resolve_database("postgresql://user:pass@host:5432/", config_mgr)
        assert result.name == "database"  # fallback name
        assert result.excluded_schemas == []

        # PostgreSQL with no path at all
        result = resolve_database("postgresql://user:pass@host:5432", config_mgr)
        assert result.name == "database"  # fallback name
        assert result.excluded_schemas == []

        # DuckDB without explicit database
        result = resolve_database("duckdb://", config_mgr)
        assert result.name == "database"
        assert result.excluded_schemas == []
