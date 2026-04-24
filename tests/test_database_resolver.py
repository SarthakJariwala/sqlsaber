"""Tests for database resolver functionality."""

from unittest.mock import Mock, patch

import pytest

from sqlsaber.config.database import DatabaseConfig
from sqlsaber.database.resolver import (
    DatabaseResolutionError,
    resolve_database,
    resolve_databases,
)


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
        assert result.type == "postgresql"
        assert result.description is None
        assert result.id is None

        # MySQL connection string
        result = resolve_database("mysql://user:pass@host:3306/mydb", config_mgr)
        assert result.name == "mydb"
        assert result.connection_string == "mysql://user:pass@host:3306/mydb"
        assert result.excluded_schemas == []
        assert result.type == "mysql"
        assert result.description is None

        # SQLite connection string
        result = resolve_database("sqlite:///test.db", config_mgr)
        assert result.name == "test"
        assert result.connection_string == "sqlite:///test.db"
        assert result.excluded_schemas == []
        assert result.type == "sqlite"
        assert result.description is None

        # CSV connection string
        result = resolve_database("csv:///data.csv", config_mgr)
        assert result.name == "data"
        assert result.connection_string == "csv:///data.csv"
        assert result.excluded_schemas == []
        assert result.type == "csv"
        assert result.description is None

        # DuckDB connection string
        result = resolve_database("duckdb:///path/to/data.duckdb", config_mgr)
        assert result.name == "data"
        assert result.connection_string == "duckdb:///path/to/data.duckdb"
        assert result.excluded_schemas == []
        assert result.type == "duckdb"
        assert result.description is None

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
        assert result.type == "csv"
        assert result.description is None

        # SQLite file
        result = resolve_database("test.db", config_mgr)
        assert result.name == "test"
        assert result.connection_string.startswith("sqlite:///")
        assert result.connection_string.endswith("test.db")
        assert result.excluded_schemas == []
        assert result.type == "sqlite"
        assert result.description is None

        # DuckDB file
        result = resolve_database("data.duckdb", config_mgr)
        assert result.name == "data"
        assert result.connection_string.startswith("duckdb:///")
        assert result.connection_string.endswith("data.duckdb")
        assert result.excluded_schemas == []
        assert result.type == "duckdb"
        assert result.description is None

    @patch("pathlib.Path.exists")
    def test_resolve_multiple_csv_file_paths(self, mock_exists):
        """Test that multiple CSV paths resolve to a combined DuckDB view connection."""
        mock_exists.return_value = True
        config_mgr = Mock()

        result = resolve_database(["a.csv", "b.csv"], config_mgr)
        assert result.connection_string.startswith("csvs:///?")
        assert "spec=" in result.connection_string
        assert result.excluded_schemas == []
        assert result.type == "csvs"
        assert result.description is None

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
        db_config.type = "postgresql"
        db_config.description = "Primary warehouse"
        db_config.to_connection_string.return_value = "postgresql://localhost:5432/mydb"
        db_config.exclude_schemas = ["foo"]
        config_mgr.get_database.return_value = db_config

        result = resolve_database("mydb", config_mgr)
        assert result.name == "mydb"
        assert result.connection_string == "postgresql://localhost:5432/mydb"
        assert result.excluded_schemas == ["foo"]
        assert result.type == "postgresql"
        assert result.description == "Primary warehouse"

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
        db_config.type = "postgresql"
        db_config.description = None
        db_config.to_connection_string.return_value = (
            "postgresql://localhost:5432/default"
        )
        db_config.exclude_schemas = ["bar"]
        config_mgr.get_default_database.return_value = db_config

        result = resolve_database(None, config_mgr)
        assert result.name == "default"
        assert result.connection_string == "postgresql://localhost:5432/default"
        assert result.excluded_schemas == ["bar"]
        assert result.type == "postgresql"
        assert result.description is None

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
        assert result.type == "postgresql"

        # PostgreSQL with no path at all
        result = resolve_database("postgresql://user:pass@host:5432", config_mgr)
        assert result.name == "database"  # fallback name
        assert result.excluded_schemas == []
        assert result.type == "postgresql"

        # DuckDB without explicit database
        result = resolve_database("duckdb://", config_mgr)
        assert result.name == "database"
        assert result.excluded_schemas == []
        assert result.type == "duckdb"

    def test_legacy_configured_database_description_defaults_to_none(self):
        """Test legacy config-like objects without description resolve cleanly."""

        class LegacyDatabaseConfig:
            name = "legacy"
            type = "mysql"
            exclude_schemas: list[str] = []

            def to_connection_string(self) -> str:
                return "mysql://localhost:3306/legacy"

        config_mgr = Mock()
        config_mgr.get_database.return_value = LegacyDatabaseConfig()

        result = resolve_database("legacy", config_mgr)
        assert result.type == "mysql"
        assert result.description is None

    @patch("pathlib.Path.exists")
    def test_resolve_databases_multiple_csvs_collapse(self, mock_exists):
        """Test multi-CSV specs collapse into one csvs database."""
        mock_exists.return_value = True
        config_mgr = Mock()

        results = resolve_databases(["a.csv", "b.csv"], config_mgr)

        assert len(results) == 1
        assert results[0].name == "a + b"
        assert results[0].type == "csvs"
        assert results[0].id == "a + b"

    def test_resolve_databases_configured_databases_in_order_with_ids(self):
        """Test multiple configured names resolve independently in order."""
        config_mgr = Mock()
        warehouse_config = Mock(spec=DatabaseConfig)
        warehouse_config.name = "warehouse"
        warehouse_config.type = "postgresql"
        warehouse_config.description = "Analytics warehouse"
        warehouse_config.to_connection_string.return_value = (
            "postgresql://localhost:5432/warehouse"
        )
        warehouse_config.exclude_schemas = ["scratch"]
        billing_config = Mock(spec=DatabaseConfig)
        billing_config.name = "billing"
        billing_config.type = "mysql"
        billing_config.description = "Billing store"
        billing_config.to_connection_string.return_value = (
            "mysql://localhost:3306/billing"
        )
        billing_config.exclude_schemas = []
        config_mgr.get_database.side_effect = [warehouse_config, billing_config]

        results = resolve_databases(["warehouse", "billing"], config_mgr)

        assert [result.name for result in results] == ["warehouse", "billing"]
        assert [result.type for result in results] == ["postgresql", "mysql"]
        assert [result.description for result in results] == [
            "Analytics warehouse",
            "Billing store",
        ]
        assert [result.id for result in results] == ["warehouse", "billing"]
        assert results[0].excluded_schemas == ["scratch"]

    def test_resolve_databases_duplicate_names_receive_suffixed_ids(self):
        """Test stable IDs are suffixed when database names duplicate."""
        config_mgr = Mock()

        results = resolve_databases(
            [
                "sqlite:///data/report.db",
                "duckdb:///warehouse/report.duckdb",
                "postgresql://localhost/report",
            ],
            config_mgr,
        )

        assert [result.name for result in results] == ["report", "report", "report"]
        assert [result.id for result in results] == ["report", "report_2", "report_3"]

    @patch("pathlib.Path.exists")
    def test_resolve_databases_mixed_csv_and_non_csv_resolve_independently(
        self, mock_exists
    ):
        """Test resolve_databases does not apply strict multi-CSV rules to mixed specs."""
        mock_exists.return_value = True
        config_mgr = Mock()

        results = resolve_databases(["a.csv", "sqlite:///test.db"], config_mgr)

        assert [result.type for result in results] == ["csv", "sqlite"]
        assert [result.id for result in results] == ["a", "test"]

    def test_resolve_databases_single_spec_assigns_id(self):
        """Test non-list specs resolve to a one-item list with an ID."""
        config_mgr = Mock()

        results = resolve_databases("sqlite:///test.db", config_mgr)

        assert len(results) == 1
        assert results[0].name == "test"
        assert results[0].id == "test"

    def test_resolve_databases_empty_list_raises(self):
        """Test empty list raises the same error as resolve_database."""
        config_mgr = Mock()

        with pytest.raises(DatabaseResolutionError, match="Empty database argument list"):
            resolve_databases([], config_mgr)
