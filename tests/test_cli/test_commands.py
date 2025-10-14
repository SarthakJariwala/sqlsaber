"""Tests for CLI commands."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from sqlsaber.cli import commands
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
        # Cyclopts prints help directly without exiting
        app(["--help"])

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
        # Cyclopts prints help directly without exiting
        app(["--help"])

        captured = capsys.readouterr()
        assert "db" in captured.out
        assert "memory" in captured.out
        assert "models" in captured.out
        assert "auth" in captured.out

    def test_maybe_configure_mlflow_no_env(self, monkeypatch):
        """MLflow stays disabled when env vars are absent."""
        monkeypatch.delenv("MLFLOW_URI", raising=False)
        monkeypatch.delenv("MLFLOW_EXP", raising=False)
        commands._MLFLOW_CONFIGURED = False
        logger = MagicMock()

        assert not commands._maybe_configure_mlflow(logger)
        logger.warning.assert_not_called()

    def test_maybe_configure_mlflow_with_env(self, monkeypatch):
        """MLflow autolog is configured when env vars and package are present."""
        monkeypatch.setenv("MLFLOW_URI", "http://localhost:5000")
        monkeypatch.setenv("MLFLOW_EXP", "sqlsaber-bench")
        commands._MLFLOW_CONFIGURED = False
        autolog_called = MagicMock()
        set_uri_called = MagicMock()
        set_exp_called = MagicMock()

        class FakeMlflow:
            def __init__(self):
                self.pydantic_ai = MagicMock(autolog=autolog_called)

            def set_tracking_uri(self, uri):
                set_uri_called(uri)

            def set_experiment(self, exp):
                set_exp_called(exp)

        monkeypatch.setitem(sys.modules, "mlflow", FakeMlflow())
        logger = MagicMock()

        assert commands._maybe_configure_mlflow(logger)
        autolog_called.assert_called_once()
        set_uri_called.assert_called_once_with("http://localhost:5000")
        set_exp_called.assert_called_once_with("sqlsaber-bench")
        logger.info.assert_called()

        # Ensure subsequent calls short-circuit
        logger.reset_mock()
        assert commands._maybe_configure_mlflow(logger)
        logger.info.assert_not_called()
        commands._MLFLOW_CONFIGURED = False
