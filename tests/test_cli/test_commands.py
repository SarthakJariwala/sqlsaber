"""Tests for CLI commands."""

import pytest

from sqlsaber.cli.commands import app


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
        assert captured.out == ""
        assert "Database connection 'nonexistent' not found" in captured.err
        assert "sqlsaber db list" in captured.err

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

    @staticmethod
    def _help_text(capsys, args: list[str]) -> str:
        with pytest.raises(SystemExit) as exc_info:
            app(args)
        assert exc_info.value.code == 0
        ascii_only = "".join(
            ch if ch.isascii() else " " for ch in capsys.readouterr().out
        )
        return " ".join(ascii_only.split())

    def test_repeated_database_help_is_not_csv_only(self, capsys):
        text = self._help_text(capsys, ["--help"])
        assert "one/more CSV files via repeated -d" not in text
        assert "multiple saved names" in text
        assert "CSV files merge" in text
        assert "-d sales -d analytics" in text
