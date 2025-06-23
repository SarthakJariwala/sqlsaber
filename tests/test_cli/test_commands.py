"""Tests for CLI commands."""

from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from sqlsaber.cli.commands import app
from sqlsaber.config.database import DatabaseConfig


class TestCLICommands:
    """Test CLI command functionality."""

    @pytest.fixture
    def runner(self):
        """Provide a CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def mock_config_manager(self):
        """Mock database config manager."""
        with patch("sqlsaber.cli.commands.config_manager") as mock:
            yield mock

    @pytest.fixture
    def mock_database_config(self):
        """Provide a mock database configuration."""
        return DatabaseConfig(
            name="test_db",
            type="postgresql",
            host="localhost",
            port=5432,
            username="user",
            password="pass",
            database="testdb",
        )

    def test_main_help(self, runner):
        """Test main help command."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "SQLSaber" in result.output
        assert "SQL assistant for your database" in result.output

    def test_query_specific_database_not_found(self, runner, mock_config_manager):
        """Test query with non-existent database name."""
        mock_config_manager.get_database.return_value = None

        result = runner.invoke(app, ["query", "-d", "nonexistent", "show tables"])

        assert result.exit_code == 1
        assert "Database connection 'nonexistent' not found" in result.output
        assert "sqlsaber db list" in result.output

    def test_subcommands_registered(self, runner):
        """Test that all subcommands are properly registered."""
        result = runner.invoke(app, ["--help"])

        assert "db" in result.output
        assert "memory" in result.output
        assert "models" in result.output
        assert "query" in result.output
