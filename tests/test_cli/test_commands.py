"""Tests for CLI commands."""

import re

import pytest

from sqlsaber.cli.commands import app


def test_query_help_mentions_multiple_databases(capsys):
    """Test query help mentions repeated database options."""
    with pytest.raises(SystemExit) as exc_info:
        app(["--help"])

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert re.search(r"multiple\W+databases", captured.out.lower())


class TestCLICommands:
    """Test CLI command functionality."""

    def test_main_help(self, capsys):
        """Test main help command."""
        with pytest.raises(SystemExit) as exc_info:
            app(["--help"])

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "SQLsaber" in captured.out
        assert "SQL assistant for your database" in captured.out
        assert "--system-prompt" in captured.out

    def test_query_specific_database_not_found(self, capsys, temp_dir, monkeypatch):
        """Test query with non-existent database name."""
        config_dir = temp_dir / "config"
        monkeypatch.setattr(
            "platformdirs.user_config_dir", lambda *args, **kwargs: str(config_dir)
        )

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
        assert "knowledge" in captured.out
        assert "models" in captured.out
        assert "auth" in captured.out
