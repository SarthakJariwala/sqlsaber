import asyncio
import subprocess
import sys
from io import StringIO

import pytest

from sqlsaber.cli import interactive as interactive_mod
from sqlsaber.cli.chat_surface import ChatSurface
from sqlsaber.cli.interactive import InteractiveSession
from sqlsaber.cli.tui_chat import build_chat_app
from sqlsaber.cli.update_check import (
    _emit_update_notice,
    bind_update_notice,
)
from sqlsaber.render import reset_io

from tests.test_cli.test_tui_chat import FakeTerminal, _fake_saber


def _started_chat_app() -> tuple:
    terminal = FakeTerminal(columns=100, rows=24)
    session = InteractiveSession(_fake_saber())
    app = build_chat_app(
        terminal=terminal,
        on_submit=lambda text: None,
        footer_text=session._footer_text(),
    )
    session.show_welcome_message(app)
    app.tui.start()
    app.tui.flush_render()
    return app, StringIO()


def _viewport(app) -> str:
    return "\n".join(app.render_plain_viewport())


def _footer_line(app) -> str:
    lines = [line for line in app.render_plain_viewport() if "DB:" in line]
    assert lines
    return lines[-1]


def test_bound_update_notice_joins_chat_above_editor_and_leaves_footer() -> None:
    app, buf = _started_chat_app()
    reset_io(stdout=buf, tty=False)
    bind_update_notice(ChatSurface(app).emit)

    _emit_update_notice()
    app.tui.flush_render()

    viewport = _viewport(app)
    assert "A new version is now available!" in viewport
    assert "uv tool update sqlsaber" in viewport
    assert "A new version is now available!" not in buf.getvalue()
    assert "uv tool update" not in _footer_line(app)
    assert "Model: gpt-test" in _footer_line(app)

    lines = app.render_plain_viewport()
    notice_at = next(
        i for i, line in enumerate(lines) if "A new version is now available!" in line
    )
    editor_rules = [
        i for i, line in enumerate(lines) if line.strip() and set(line.strip()) <= {"─"}
    ]
    assert editor_rules
    assert notice_at < editor_rules[0]


def test_notice_before_bind_flushes_into_chat_not_stdout() -> None:
    app, buf = _started_chat_app()
    reset_io(stdout=buf, tty=False)

    _emit_update_notice()
    assert "A new version is now available!" not in buf.getvalue()
    assert "A new version is now available!" not in _viewport(app)

    bind_update_notice(ChatSurface(app).emit)
    app.tui.flush_render()

    viewport = _viewport(app)
    assert "A new version is now available!" in viewport
    assert "uv tool update sqlsaber" in viewport
    assert "A new version is now available!" not in buf.getvalue()
    assert "Model: gpt-test" in _footer_line(app)


def test_update_check_without_setup_logging_does_not_print_to_stdout() -> None:
    code = """
import asyncio
from unittest.mock import patch

async def fake_fetch():
    return "99.0.0"

async def main() -> None:
    from sqlsaber.cli.update_check import (
        _check_and_notify,
        bind_update_notice,
        reset_update_check,
    )

    reset_update_check()
    emitted: list[str] = []
    bind_update_notice(lambda *blocks: emitted.append("yes"))
    with (
        patch("sqlsaber.cli.update_check._get_current_version", lambda: "0.1.0"),
        patch("sqlsaber.cli.update_check._fetch_latest_version", fake_fetch),
    ):
        await _check_and_notify()
    if emitted != ["yes"]:
        raise SystemExit(f"notice missing: {emitted!r}")

asyncio.run(main())
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "update_check.available" not in result.stdout
    assert "update_check.available" not in result.stderr


def test_one_shot_bind_still_prints_to_stdout() -> None:
    from sqlsaber.cli.output import out

    buf = StringIO()
    reset_io(stdout=buf, tty=False)
    bind_update_notice(out)
    _emit_update_notice()
    text = buf.getvalue()
    assert "A new version is now available!" in text
    assert "uv tool update sqlsaber" in text


@pytest.mark.asyncio
async def test_interactive_session_binds_update_notice_before_tui_start(
    monkeypatch,
) -> None:
    bound: list[object] = []

    def fake_bind(emit: object) -> None:
        bound.append(emit)

    monkeypatch.setattr(interactive_mod, "bind_update_notice", fake_bind, raising=False)

    terminal = FakeTerminal(columns=80, rows=12)
    session = InteractiveSession.__new__(InteractiveSession)
    session.log = type("FakeLog", (), {"info": lambda *a, **k: None})()
    session.saber = _fake_saber(database_names=("test",))
    session.autocomplete_provider = None
    session._handoff_mode = False
    session.current_task = None
    session._submit_pending = False
    session._exit_finalized = False
    session.usage = interactive_mod.UsageMeter(
        model_id=session._model_id, on_change=session._refresh_footer
    )
    session.before_prompt_loop = lambda: asyncio.sleep(0)
    session._load_history = lambda: []
    session.show_welcome_message = lambda app: None
    session._finalize_exit = lambda: asyncio.sleep(0)

    def fake_build_chat_app(**kwargs):
        app = build_chat_app(terminal=terminal, **kwargs)
        asyncio.get_running_loop().call_soon(app.stop)
        return app

    monkeypatch.setattr(interactive_mod, "build_chat_app", fake_build_chat_app)
    await session.run()
    assert bound, "InteractiveSession.run must bind the chat surface before tui.start"
