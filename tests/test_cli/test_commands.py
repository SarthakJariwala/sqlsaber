"""Tests for CLI commands."""

from unittest.mock import patch

import pytest

from sqlsaber.cli.commands import app
from sqlsaber.config.database import DatabaseConfig


class TestCLICommands:
    """Test CLI command functionality."""

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

    def test_main_help(self, capsys):
        """Test main help command."""
        with pytest.raises(SystemExit) as exc_info:
            app(["--help"])

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "SQLsaber" in captured.out
        assert "SQL assistant for your database" in captured.out

    def test_query_specific_database_not_found(self, capsys, mock_config_manager):
        """Test query with non-existent database name."""
        mock_config_manager.get_database.return_value = None

        with pytest.raises(SystemExit) as exc_info:
            app(["-d", "nonexistent", "show tables"])

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Database connection 'nonexistent' not found" in captured.out
        assert "sqlsaber db list" in captured.out

    def test_subcommands_registered(self, capsys):
        """Test that all subcommands are properly registered."""
        with pytest.raises(SystemExit) as exc_info:
            app(["--help"])

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "db" in captured.out
        assert "memory" in captured.out
        assert "models" in captured.out
        assert "auth" in captured.out
