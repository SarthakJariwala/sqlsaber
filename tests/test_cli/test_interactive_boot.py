"""Interactive TUI boot paints before pydantic-ai import."""

from __future__ import annotations

import subprocess
import sys

import pytest

from sqlsaber.cli.interactive import InteractiveSession

from tests.test_cli.test_tui_chat import FakeTerminal


def test_commands_import_does_not_load_structlog_or_httpx() -> None:
    code = """
import sys
import sqlsaber.cli.commands  # noqa: F401

loaded = [
    name
    for name in (
        "structlog",
        "httpx",
        "keyring",
        "pydantic_ai",
        "sqlsaber.cli.auth",
        "sqlsaber.cli.models",
        "sqlsaber.cli.update_check",
        "sqlsaber.config.logging",
    )
    if name in sys.modules
]
assert not loaded, loaded
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_interactive_query_does_not_import_pydantic_ai_before_first_paint() -> None:
    """Retention pulls pydantic-ai; it must stay off the first-paint path."""
    code = """
import sys
from unittest.mock import patch

class Probe(Exception):
    pass

class FakeSession:
    @classmethod
    def start_unbound_shell(cls, **kwargs):
        loaded = [
            name
            for name in (
                "pydantic_ai",
                "sqlsaber.cli.retention",
                "sqlsaber.threads.storage",
                "sqlsaber.sdk.client",
                "structlog",
                "httpx",
                "sqlsaber.config.logging",
                "sqlsaber.cli.update_check",
            )
            if name in sys.modules
        ]
        raise Probe(",".join(loaded) or "clean")

from sqlsaber.cli.commands import query

with (
    patch("sqlsaber.cli.interactive.InteractiveSession", FakeSession),
    patch("sqlsaber.cli.commands.schedule_update_check"),
):
    try:
        query(database=["analytics"])
    except Probe as exc:
        assert str(exc) == "clean", str(exc)
    else:
        raise SystemExit("probe not raised")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_interactive_module_import_does_not_load_pydantic_ai() -> None:
    code = """
import sys
from sqlsaber.cli.interactive import InteractiveSession, ChatShell

assert "pydantic_ai" not in sys.modules
assert "sqlsaber.sdk.client" not in sys.modules
assert "sqlsaber.cli.usage" not in sys.modules
assert "sqlsaber.cli.tui_streaming" not in sys.modules
assert "sqlsaber.cli.stream_presenter" not in sys.modules
assert InteractiveSession.__name__ == "InteractiveSession"
assert ChatShell.__name__ == "ChatShell"
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_boot_footer_includes_db_ready_needle() -> None:
    assert "DB:" in InteractiveSession.boot_footer("verification.db")
    assert "verification" in InteractiveSession.boot_footer(
        "/tmp/fixtures/verification.db"
    )
    assert "DB:" in InteractiveSession.boot_footer(None)


@pytest.mark.asyncio
async def test_start_unbound_shell_paints_slash_hint_and_db_footer() -> None:
    terminal = FakeTerminal(columns=100, rows=24)
    shell = InteractiveSession.start_unbound_shell(
        database="/tmp/fixtures/verification.db",
        terminal=terminal,
    )
    try:
        shell.app.tui.flush_render()
        text = "\n".join(shell.app.render_plain_viewport())
        folded = text.casefold()
        assert "slash commands" in folded
        assert "table name completions" in folded
        assert "DB:" in text
        assert "verification" in folded
        assert terminal.started is True
    finally:
        shell.stop()
    assert terminal.stopped is True


@pytest.mark.asyncio
async def test_unbound_shell_slash_opens_palette() -> None:
    terminal = FakeTerminal(columns=100, rows=24)
    shell = InteractiveSession.start_unbound_shell(
        database="verification.db",
        terminal=terminal,
    )
    try:
        terminal.send_input("/")
        shell.app.tui.flush_render()
        text = "\n".join(shell.app.render_plain_viewport())
        assert shell.app.is_command_palette_open() is True
        assert "Thinking mode" in text
        assert "Command help" in text
    finally:
        shell.stop()
