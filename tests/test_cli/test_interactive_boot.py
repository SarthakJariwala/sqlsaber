"""Interactive TUI boot paints before pydantic-ai import."""

from __future__ import annotations

import asyncio
import subprocess
import sys
import threading

import pytest

from sqlsaber.cli.interactive import InteractiveSession, _signal_loop_event

from tests.test_cli.test_tui_chat import FakeTerminal, _fake_saber


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


def test_preview_footer_does_not_import_pydantic_ai() -> None:
    code = """
import sys
from unittest.mock import MagicMock

from sqlsaber.cli.interactive import InteractiveSession

shell = MagicMock()
shell.app.set_footer(InteractiveSession.preview_footer(None))
loaded = [
    name
    for name in (
        "pydantic_ai",
        "sqlsaber.sdk.client",
        "sqlsaber.threads.storage",
        "sqlsaber.cli.retention",
    )
    if name in sys.modules
]
assert not loaded, loaded
shell.app.set_footer.assert_called_once()
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_preview_footer_uses_saved_default_without_sqlsaber(monkeypatch) -> None:
    from types import SimpleNamespace

    monkeypatch.setattr(
        "sqlsaber.config.database.DatabaseConfigManager.get_default_database",
        lambda self: SimpleNamespace(name="verification", type="sqlite"),
    )

    def fake_default(cls=None):
        return SimpleNamespace(
            model=SimpleNamespace(
                name="openai:gpt-test",
                thinking_enabled=False,
                thinking_level=SimpleNamespace(value="medium"),
            )
        )

    monkeypatch.setattr(
        "sqlsaber.config.settings.Config.default",
        classmethod(fake_default),
    )
    text = InteractiveSession.preview_footer(None)
    assert "DB: verification (sqlite)" in text
    assert "Model: openai:gpt-test" in text
    assert "Thinking: off" in text


@pytest.mark.asyncio
async def test_bind_signal_from_other_thread_wakes_wait() -> None:
    terminal = FakeTerminal(columns=100, rows=24)
    shell = InteractiveSession.start_unbound_shell(
        database=None,
        terminal=terminal,
    )
    try:
        waiter = asyncio.create_task(shell.wait_for_bind_or_exit())
        await asyncio.sleep(0)
        threading.Thread(
            target=_signal_loop_event,
            args=(shell.loop, shell.bind_event),
            daemon=True,
        ).start()
        assert await asyncio.wait_for(waiter, timeout=2) is True
    finally:
        shell.stop()


@pytest.mark.asyncio
async def test_wait_for_bind_or_exit_skips_bind_when_already_exited() -> None:
    terminal = FakeTerminal(columns=100, rows=24)
    shell = InteractiveSession.start_unbound_shell(
        database=None,
        terminal=terminal,
    )
    try:
        shell.exit_event.set()
        assert await shell.wait_for_bind_or_exit() is False
        shell.bind_event.set()
        assert await shell.wait_for_bind_or_exit() is False
    finally:
        shell.stop()


@pytest.mark.asyncio
async def test_unbound_exit_command_does_not_request_bind() -> None:
    terminal = FakeTerminal(columns=100, rows=24)
    shell = InteractiveSession.start_unbound_shell(
        database=None,
        terminal=terminal,
    )
    try:
        for char in "exit":
            terminal.send_input(char)
        terminal.send_input("\r")
        shell.app.tui.flush_render()
        assert shell.bind_event.is_set() is False
        assert shell.exit_event.is_set() is True
        assert await shell.wait_for_bind_or_exit() is False
    finally:
        shell.stop()


@pytest.mark.asyncio
async def test_unbound_ctrl_c_during_starting_exits_without_bind() -> None:
    terminal = FakeTerminal(columns=100, rows=24)
    shell = InteractiveSession.start_unbound_shell(
        database=None,
        terminal=terminal,
    )
    try:
        shell.app.submit("count rows")
        shell.app.tui.flush_render()
        assert shell.bind_event.is_set() is True
        terminal.send_input("\x03")
        shell.app.tui.flush_render()
        assert shell.exit_event.is_set() is True
        assert await shell.wait_for_bind_or_exit() is False
    finally:
        shell.stop()


@pytest.mark.asyncio
async def test_unbound_clear_does_not_request_bind() -> None:
    terminal = FakeTerminal(columns=100, rows=24)
    shell = InteractiveSession.start_unbound_shell(
        database=None,
        terminal=terminal,
    )
    try:
        shell.app.submit("/clear")
        shell.app.tui.flush_render()
        assert shell.bind_event.is_set() is False
        assert shell.exit_event.is_set() is False
        text = "\n".join(shell.app.render_plain_viewport())
        assert "Conversation history cleared." in text
    finally:
        shell.stop()


@pytest.mark.asyncio
async def test_unbound_submit_requests_bind_and_keeps_text() -> None:
    terminal = FakeTerminal(columns=100, rows=24)
    shell = InteractiveSession.start_unbound_shell(
        database=None,
        terminal=terminal,
    )
    try:
        for char in "count rows":
            terminal.send_input(char)
        terminal.send_input("\r")
        shell.app.tui.flush_render()
        assert shell.bind_event.is_set() is True
        assert shell.queued["query"] == "count rows"
        assert shell.app.editor.get_text() == "count rows"
        assert await shell.wait_for_bind_or_exit() is True
    finally:
        shell.stop()
    assert "DB:" in InteractiveSession.boot_footer("verification.db")
    assert "verification" in InteractiveSession.boot_footer(
        "/tmp/fixtures/verification.db"
    )
    assert "DB:" in InteractiveSession.boot_footer(None)


@pytest.mark.asyncio
async def test_run_submits_queued_query_after_bind() -> None:
    terminal = FakeTerminal(columns=100, rows=24)
    shell = InteractiveSession.start_unbound_shell(
        database=None,
        terminal=terminal,
    )
    try:
        for char in "count rows":
            terminal.send_input(char)
        terminal.send_input("\r")
        shell.app.tui.flush_render()

        submitted: list[str] = []
        session = InteractiveSession.__new__(InteractiveSession)
        session.log = type("FakeLog", (), {"info": lambda *a, **k: None})()
        session.saber = _fake_saber()
        session.autocomplete_provider = None
        session._handoff_mode = False
        session._submit_pending = False
        session.current_task = None
        session._exit_finalized = False
        session.streaming_handler = None
        session.before_prompt_loop = lambda: asyncio.sleep(0)
        session._load_history = lambda: []
        session._footer_text = lambda: "DB: test"
        session._create_streaming_handler = lambda app: None
        session._finalize_exit = lambda: asyncio.sleep(0)

        def capture_queue(app, surface, user_query, *, loop):
            del app, surface, loop
            submitted.append(user_query)
            shell.exit_event.set()
            return True

        session._queue_submit = capture_queue
        await session.run(shell=shell)
        assert submitted == ["count rows"]
        text = "\n".join(shell.app.render_plain_viewport())
        assert "count rows" in text
        assert shell.app.editor.get_text() == ""
    finally:
        shell.stop()


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
        assert "Welcome to SQLsaber!" in text
        assert "slash commands" in folded
        assert "table name completions" in folded
        assert "DB:" in text
        assert "verification" in folded
        assert "█" not in text
        assert terminal.started is True
    finally:
        shell.stop()
    assert terminal.stopped is True


@pytest.mark.asyncio
async def test_unbound_editor_accepts_text_before_session_bind() -> None:
    terminal = FakeTerminal(columns=100, rows=24)
    shell = InteractiveSession.start_unbound_shell(
        database=None,
        terminal=terminal,
    )
    try:
        for char in "hello":
            terminal.send_input(char)
        shell.app.tui.flush_render()
        assert shell.app.editor.get_text() == "hello"
        assert "session" not in shell.session_slot
        text = "\n".join(shell.app.render_plain_viewport())
        assert "hello" in text
        assert "DB: starting..." in text
    finally:
        shell.stop()


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
