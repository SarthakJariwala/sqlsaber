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
        assert "--csv-tool-results" in captured.out

    @pytest.mark.parametrize("prefix", [[], ["threads", "resume", "test-thread"]])
    @pytest.mark.parametrize(
        "flag,expected",
        [(None, False), ("--csv-tool-results", True), ("--no-csv-tool-results", False)],
    )
    def test_csv_flag_parsing(self, prefix, flag, expected):
        _, bound, _ = app.parse_args([*prefix, *([flag] if flag else [])])
        bound.apply_defaults()
        assert bound.arguments["csv_tool_results"] is expected

    @pytest.mark.parametrize("interactive", [False, True])
    @pytest.mark.parametrize("csv_tool_results", [False, True])
    def test_query_passes_csv_choice_to_session(
        self, monkeypatch, interactive, csv_tool_results
    ):
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from sqlsaber.cli import commands
        from sqlsaber.cli.interactive import ChatShell, InteractiveSession

        class SessionReached(Exception):
            pass

        create = AsyncMock(side_effect=SessionReached)

        def start_unbound_shell(**kwargs):
            loop = asyncio.get_running_loop()
            bind_event = asyncio.Event()
            bind_event.set()
            app = MagicMock()
            app.tui.stopped = True
            return ChatShell(
                app=app,
                session_slot={},
                exit_event=asyncio.Event(),
                loop=loop,
                bind_event=bind_event,
            )

        monkeypatch.setattr(commands, "_create_cli_saber", create)
        monkeypatch.setattr(commands, "needs_onboarding", lambda _: False)
        monkeypatch.setattr(commands, "schedule_update_check", lambda: None)
        monkeypatch.setattr(commands, "_ensure_logging", MagicMock())
        monkeypatch.setattr(commands.sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr(
            InteractiveSession, "start_unbound_shell", start_unbound_shell
        )
        monkeypatch.setattr(
            InteractiveSession, "preview_footer", lambda *a, **k: "DB: test"
        )
        with pytest.raises(SessionReached):
            commands.query(
                None if interactive else "Show tables",
                csv_tool_results=csv_tool_results,
            )
        assert create.await_args.kwargs["csv_tool_results"] is csv_tool_results

    def test_interactive_immediate_exit_does_not_construct_sqlsaber(
        self, monkeypatch
    ) -> None:
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from sqlsaber.cli import commands
        from sqlsaber.cli.interactive import ChatShell, InteractiveSession

        created = False

        async def create(**kwargs):
            nonlocal created
            created = True
            raise AssertionError(
                "interactive exit before a query must not construct SQLSaber"
            )

        def start_unbound_shell(**kwargs):
            loop = asyncio.get_running_loop()
            exit_event = asyncio.Event()
            exit_event.set()
            app = MagicMock()
            app.tui.stopped = True
            return ChatShell(
                app=app,
                session_slot={},
                exit_event=exit_event,
                loop=loop,
            )

        monkeypatch.setattr(
            commands, "_create_cli_saber", AsyncMock(side_effect=create)
        )
        monkeypatch.setattr(commands, "needs_onboarding", lambda _: False)
        scheduled: list[bool] = []
        monkeypatch.setattr(
            commands, "schedule_update_check", lambda: scheduled.append(True)
        )
        monkeypatch.setattr(commands, "_ensure_logging", MagicMock())
        monkeypatch.setattr(commands.sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr(
            InteractiveSession, "start_unbound_shell", start_unbound_shell
        )
        monkeypatch.setattr(
            InteractiveSession, "preview_footer", lambda *a, **k: "DB: test"
        )

        commands.query(None)

        assert created is False
        assert scheduled == [True]

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
        assert "CSV/Parquet files merge" in text
        assert "-d sales -d analytics" in text
