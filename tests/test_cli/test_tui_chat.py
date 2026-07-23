import asyncio
import time
from collections.abc import Callable
from io import BytesIO, StringIO
from types import SimpleNamespace

import pytest
import saber_tui.utils as tui_utils
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelResponse,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
    ToolReturnPart,
)
from pydantic_ai.usage import RequestUsage, RunUsage
from rich.table import Table
from saber_tui import PosixProcessTerminal, TerminalCapabilities, WindowsProcessTerminal
from saber_tui.components import Box
from saber_tui.components import Image as TUIImage
from saber_tui.components import Markdown
from saber_tui.stdin_buffer import StdinBuffer
from saber_tui.utils import strip_ansi, visible_width

import sqlsaber.cli.interactive as interactive
from sqlsaber.cli import tui_chat
from sqlsaber.cli.interactive import InteractiveSession
from sqlsaber.cli.tui_chat import ChatApp, build_chat_app
from sqlsaber.cli.tui_streaming import TUIStreamingQueryHandler
from sqlsaber.config.settings import ThinkingLevel
from sqlsaber.theme.manager import create_console, get_theme_manager


class FakeTerminal:
    def __init__(self, columns: int = 100, rows: int = 16) -> None:
        self._columns = columns
        self._rows = rows
        self._on_input: Callable[[str], None] | None = None
        self._on_resize: Callable[[], None] | None = None
        self.writes: list[str] = []
        self.started = False
        self.stopped = False

    def start(
        self, on_input: Callable[[str], None], on_resize: Callable[[], None]
    ) -> None:
        self._on_input = on_input
        self._on_resize = on_resize
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def drain_input(self, max_ms: int = 1000, idle_ms: int = 50) -> None:
        _ = max_ms, idle_ms

    def write(self, data: str) -> None:
        self.writes.append(data)

    @property
    def columns(self) -> int:
        return self._columns

    @property
    def rows(self) -> int:
        return self._rows

    @property
    def kitty_protocol_active(self) -> bool:
        return False

    def move_by(self, lines: int) -> None:
        _ = lines

    def hide_cursor(self) -> None:
        self.write("\x1b[?25l")

    def show_cursor(self) -> None:
        self.write("\x1b[?25h")

    def clear_line(self) -> None:
        self.write("\x1b[K")

    def clear_from_cursor(self) -> None:
        self.write("\x1b[0J")

    def clear_screen(self) -> None:
        self.write("\x1b[2J\x1b[H")

    def set_title(self, title: str) -> None:
        _ = title

    def set_progress(self, active: bool) -> None:
        _ = active

    def send_input(self, data: str) -> None:
        assert self._on_input is not None
        self._on_input(data)


def test_chat_app_keeps_editor_mounted_and_focused_after_submit() -> None:
    terminal = FakeTerminal()
    submitted: list[str] = []
    app = build_chat_app(terminal=terminal, on_submit=submitted.append)
    app.tui.start()

    assert app.editor.focused is True
    terminal.send_input("h")
    terminal.send_input("i")
    terminal.send_input("\r")
    app.tui.flush_render()

    assert submitted == ["hi"]
    assert app.editor.focused is True
    assert app.editor.get_text() == ""
    assert app.editor.history == ["hi"]


def test_chat_app_can_route_empty_submit_when_allowed() -> None:
    terminal = FakeTerminal()
    submitted: list[str] = []
    app = build_chat_app(
        terminal=terminal,
        on_submit=submitted.append,
        should_submit_empty=lambda: True,
    )
    app.tui.start()

    terminal.send_input("\r")

    assert submitted == [""]
    assert app.editor.get_text() == ""


def test_chat_app_keeps_rejected_submission_in_editor_without_echoing() -> None:
    terminal = FakeTerminal()
    app = build_chat_app(terminal=terminal, on_submit=lambda text: False)
    app.tui.start()

    for char in "still running":
        terminal.send_input(char)
    terminal.send_input("\r")

    assert app.editor.get_text() == "still running"
    assert app.editor.history == []
    assert app.chat_container.children == []


def test_chat_app_expands_saber_tui_width_cache(monkeypatch) -> None:
    monkeypatch.setattr(tui_utils, "_WIDTH_CACHE_SIZE", 512)
    tui_utils._width_cache.clear()

    build_chat_app(terminal=FakeTerminal(), on_submit=lambda text: None)

    assert tui_utils._WIDTH_CACHE_SIZE >= 65_536


def test_chat_app_uses_multiline_editor_before_submit() -> None:
    terminal = FakeTerminal()
    submitted: list[str] = []
    app = build_chat_app(terminal=terminal, on_submit=submitted.append)
    app.tui.start()

    for char in "select *":
        terminal.send_input(char)
    terminal.send_input("\x1b[13;2u")
    for char in "from users":
        terminal.send_input(char)
    terminal.send_input("\r")

    assert submitted == ["select *\nfrom users"]
    assert app.editor.focused is True


@pytest.mark.parametrize("shift_enter", ["\n", "\x1b\r"])
def test_chat_app_accepts_non_kitty_shift_enter_sequences(shift_enter: str) -> None:
    terminal = FakeTerminal()
    submitted: list[str] = []
    app = build_chat_app(terminal=terminal, on_submit=submitted.append)
    app.tui.start()

    terminal.send_input("h")
    terminal.send_input(shift_enter)
    terminal.send_input("i")

    assert submitted == []
    assert app.editor.get_text() == "h\ni"

    terminal.send_input("\r")

    assert submitted == ["h\ni"]


def test_user_messages_render_as_padded_background_blocks_without_role_label() -> None:
    terminal = FakeTerminal(columns=40, rows=12)
    app = build_chat_app(terminal=terminal, on_submit=lambda text: None)
    app.tui.start()

    app.submit("show revenue by month")
    app.tui.flush_render()

    raw_lines = app.chat_container.render(terminal.columns)
    plain_lines = [strip_ansi(line) for line in raw_lines]
    raw_chat = "\n".join(raw_lines)
    viewport = "\n".join(app.render_plain_viewport())

    assert plain_lines[0] == " " * terminal.columns
    assert plain_lines[1].startswith("show revenue by month")
    assert not plain_lines[1].startswith(" ")
    assert plain_lines[2] == " " * terminal.columns
    assert "\x1b[48;2;" in raw_chat
    assert "show revenue by month" in viewport
    assert "You:" not in viewport


def test_user_message_colors_follow_active_theme(monkeypatch) -> None:
    monkeypatch.setenv("SQLSABER_THEME", "dracula")
    get_theme_manager.cache_clear()
    try:
        terminal = FakeTerminal(columns=40, rows=12)
        app = build_chat_app(terminal=terminal, on_submit=lambda text: None)
        app.tui.start()

        app.submit("theme check")
        app.tui.flush_render()

        raw_chat = "\n".join(app.chat_container.render(terminal.columns))
        assert "\x1b[38;2;80;250;123m" in raw_chat
        assert "\x1b[48;2;68;71;90m" in raw_chat
        assert "\x1b[38;2;125;211;252m" not in raw_chat
    finally:
        get_theme_manager.cache_clear()


def test_native_markdown_colors_follow_active_theme(monkeypatch) -> None:
    monkeypatch.setenv("SQLSABER_THEME", "dracula")
    get_theme_manager.cache_clear()
    try:
        app = build_chat_app(
            terminal=FakeTerminal(columns=60), on_submit=lambda text: None
        )

        component = app.append_markdown("# Themed heading\n\n`SELECT`")
        rendered = "\n".join(component.render(60))

        assert isinstance(component, Markdown)
        assert "\x1b[38;2;255;121;198m" in rendered
        assert "\x1b[38;2;189;147;249m" in rendered
    finally:
        get_theme_manager.cache_clear()


def test_status_uses_cancellable_loader_without_stealing_editor_focus() -> None:
    terminal = FakeTerminal(columns=80, rows=12)
    cancelled: list[bool] = []
    app = build_chat_app(
        terminal=terminal,
        on_submit=lambda text: None,
        on_cancel=lambda: cancelled.append(True),
    )
    app.tui.start()

    app.set_loading("Crunching data...")
    app.tui.flush_render()
    viewport = "\n".join(app.render_plain_viewport())

    assert "Crunching data..." in viewport
    assert app.status.loader is not None
    assert app.editor.focused is True

    terminal.send_input("\x03")

    assert cancelled == [True]
    assert app.editor.focused is True
    app.clear_status()


def test_status_and_footer_render_within_narrow_terminal_width() -> None:
    terminal = FakeTerminal(columns=24, rows=8)
    app = build_chat_app(
        terminal=terminal,
        on_submit=lambda text: None,
        footer_text="DB: very_long_database_name_that_exceeds_width | Model: very-long-model-name",
    )
    app.tui.start()
    app.set_status("Type the handoff goal and press Enter.")

    lines, _ = app.tui._prepare_lines(terminal.columns, terminal.rows)

    assert all(visible_width(line) <= terminal.columns for line in lines)


def test_bare_slash_opens_command_palette_without_editor_autocomplete() -> None:
    terminal = FakeTerminal(columns=80, rows=18)
    changes: list[tuple[bool, ThinkingLevel | None]] = []

    def open_palette(app: ChatApp) -> None:
        app.show_command_palette(
            thinking_enabled=False,
            thinking_level=ThinkingLevel.MEDIUM,
            on_thinking_change=lambda enabled, level: changes.append((enabled, level)),
        )

    app = build_chat_app(
        terminal=terminal,
        on_submit=lambda text: None,
        on_open_command_palette=open_palette,
        footer_text="DB: ocean-local (PostgreSQL) | Model: claude-opus-4-6",
    )
    app.tui.start()

    terminal.send_input("/")
    app.tui.flush_render()

    viewport = "\n".join(app.render_plain_viewport())
    assert app.editor.focused is False
    assert app.editor.get_text() == ""
    assert app.editor.autocomplete_list is None
    assert app.chat_container.children == []
    assert app.tui.has_overlay() is False
    assert "Command Palette" not in viewport
    assert "Thinking mode" in viewport
    assert "off" in viewport
    assert "DB: ocean-local" in viewport
    assert all(
        "run" not in line
        for line in app.render_plain_viewport()
        if "Handoff thread" in line or "Clear conversation" in line or "Exit" in line
    )
    assert all(
        "─" not in line
        for line in app.render_plain_viewport()
        if "Command Palette" in line or ">" in line or "Thinking mode" in line
    )
    raw_viewport = "\n".join(app.tui._prepare_lines(terminal.columns, terminal.rows)[0])
    assert raw_viewport.count("\x1b[7m") == 1
    assert "\x1b[48;2;" not in raw_viewport

    terminal.send_input("\r")

    assert changes == [(True, ThinkingLevel.MINIMAL)]

    terminal.send_input("\x1b")

    assert app.tui.has_overlay() is False
    assert app.editor.focused is True


def test_status_snapshots_loader_across_render_cancel_and_dispose() -> None:
    app = build_chat_app(terminal=FakeTerminal(columns=80), on_submit=lambda text: None)

    class RacingLoader:
        def render(self, width: int) -> list[str]:
            app.status.loader = None
            return ["loading".ljust(width)]

        def handle_input(self, data: str) -> None:
            assert data == "\x03"
            app.status.loader = None

        def dispose(self) -> None:
            assert app.status.loader is None

    loader = RacingLoader()
    app.status.loader = loader
    assert app.status.render(80)[0].startswith("loading")

    app.status.loader = loader
    assert app.status.cancel_loading() is True

    app.status.loader = loader
    app.status.dispose()
    assert app.status.loader is None


def test_command_palette_ctrl_c_closes_without_running_cancel_handler() -> None:
    terminal = FakeTerminal(columns=80, rows=18)
    cancelled: list[bool] = []

    def open_palette(app: ChatApp) -> None:
        app.show_command_palette(
            thinking_enabled=False,
            thinking_level=ThinkingLevel.MEDIUM,
            on_thinking_change=lambda _enabled, _level: None,
        )

    app = build_chat_app(
        terminal=terminal,
        on_submit=lambda text: None,
        on_cancel=lambda: cancelled.append(True),
        on_open_command_palette=open_palette,
    )
    app.tui.start()

    terminal.send_input("/")
    terminal.send_input("\x03")

    assert cancelled == []
    assert app.editor.focused is True
    assert "Thinking mode" not in "\n".join(app.render_plain_viewport())


def test_build_chat_app_uses_posix_process_terminal_by_default_on_non_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("platform.system", lambda: "Darwin")

    app = build_chat_app(on_submit=lambda text: None)

    assert type(app.tui.terminal) is PosixProcessTerminal


def test_build_chat_app_uses_windows_process_terminal_by_default_on_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("platform.system", lambda: "Windows")

    app = build_chat_app(on_submit=lambda text: None)

    assert type(app.tui.terminal) is WindowsProcessTerminal


def test_saber_tui_stdin_buffer_emits_standalone_escape_after_timeout() -> None:
    emitted: list[str] = []
    stdin_buffer = StdinBuffer(on_data=emitted.append, timeout=0.001)

    try:
        stdin_buffer.process("\x1b")

        deadline = time.monotonic() + 0.1
        while not emitted and time.monotonic() < deadline:
            time.sleep(0.005)

        assert emitted == ["\x1b"]
    finally:
        stdin_buffer.destroy()


def test_slash_command_submit_runs_silently_without_echoing_user_message() -> None:
    terminal = FakeTerminal(columns=80, rows=18)
    submitted: list[str] = []
    app = build_chat_app(terminal=terminal, on_submit=submitted.append)
    app.tui.start()

    app.submit("/clear")

    assert submitted == ["/clear"]
    assert app.editor.get_text() == ""
    assert app.editor.history == []
    assert app.chat_container.children == []


def test_ctrl_d_while_loading_cancels_without_stopping_tui() -> None:
    terminal = FakeTerminal(columns=80, rows=12)
    cancelled: list[bool] = []
    exited: list[bool] = []
    app = build_chat_app(
        terminal=terminal,
        on_submit=lambda text: None,
        on_cancel=lambda: cancelled.append(True),
        on_exit=lambda: exited.append(True),
    )
    app.tui.start()
    app.set_loading("Crunching data...")

    terminal.send_input("\x04")

    assert cancelled == [True]
    assert exited == []
    assert terminal.stopped is False
    assert app.editor.focused is True
    app.clear_status()


def test_status_keeps_idle_spacer_before_editor() -> None:
    terminal = FakeTerminal(columns=80, rows=12)
    app = build_chat_app(terminal=terminal, on_submit=lambda text: None)
    app.tui.start()

    assert app.status.render(terminal.columns) == [" " * terminal.columns]

    app.set_loading("Crunching data...")
    app.clear_status()

    assert app.status.render(terminal.columns) == [" " * terminal.columns]


def test_chat_app_renders_footer_text() -> None:
    terminal = FakeTerminal(columns=80, rows=12)
    app = build_chat_app(
        terminal=terminal,
        on_submit=lambda text: None,
        footer_text="DB: analytics (DuckDB) | Model: gpt-test",
    )
    app.tui.start()
    app.tui.flush_render()

    viewport = "\n".join(app.render_plain_viewport())
    assert "DB: analytics (DuckDB)" in viewport
    assert "Model: gpt-test" in viewport


def test_interactive_footer_includes_usage_cost_and_context() -> None:
    session = InteractiveSession.__new__(InteractiveSession)
    session.database_name = "analytics"
    session._db_type_name = lambda: "DuckDB"
    session.sqlsaber_agent = SimpleNamespace(
        agent=SimpleNamespace(model=SimpleNamespace(model_name="gpt-test")),
    )
    session.session_usage = interactive.SessionUsage(
        total_input_tokens=4200,
        total_output_tokens=820,
        current_context_tokens=999,
        total_cost_usd=0.0123,
    )

    footer = session._footer_text()

    assert "DB: analytics (DuckDB)" in footer
    assert "Model: gpt-test" in footer
    assert "Usage: ↑4.2k ↓820" in footer
    assert "Ctx: 999" in footer
    assert "Cost: $0.0123" in footer


def test_interactive_footer_includes_dangerous_mode_when_enabled() -> None:
    session = InteractiveSession.__new__(InteractiveSession)
    session.database_name = "analytics"
    session._db_type_name = lambda: "DuckDB"
    session.allow_dangerous = True
    session.sqlsaber_agent = SimpleNamespace(
        agent=SimpleNamespace(model=SimpleNamespace(model_name="gpt-test")),
    )
    session.session_usage = interactive.SessionUsage()

    footer = session._footer_text()

    assert tui_chat.DANGEROUS_MODE_FOOTER_LABEL in footer


def test_interactive_footer_shows_all_database_names() -> None:
    session = InteractiveSession.__new__(InteractiveSession)
    session.database_name = "prod"
    session.database_names = ["prod", "staging", "warehouse"]
    session._db_type_name = lambda: "PostgreSQL"
    session.sqlsaber_agent = SimpleNamespace(
        agent=SimpleNamespace(model=SimpleNamespace(model_name="gpt-test")),
    )
    session.session_usage = interactive.SessionUsage()

    footer = session._footer_text()

    assert "DBs: prod, staging, warehouse" in footer
    assert "DB: prod (PostgreSQL)" not in footer
    assert "Model: gpt-test" in footer


@pytest.mark.asyncio
async def test_execute_query_refreshes_footer_usage_cost_and_context() -> None:
    terminal = FakeTerminal(columns=160, rows=12)
    app = build_chat_app(
        terminal=terminal,
        on_submit=lambda text: None,
        footer_text="DB: analytics (DuckDB) | Model: claude-sonnet-4-5 | Usage: ↑0 ↓0 | Ctx: 0 | Cost: $0.0000",
    )
    app.tui.start()

    session = InteractiveSession.__new__(InteractiveSession)
    session.log = SimpleNamespace(info=lambda *args, **kwargs: None)
    session.database_name = "analytics"
    session._db_type_name = lambda: "DuckDB"
    session.session_usage = interactive.SessionUsage()
    session.message_history = []
    session.current_task = None
    session.cancellation_token = None
    session.sqlsaber_agent = SimpleNamespace(
        agent=SimpleNamespace(
            model=SimpleNamespace(
                model_name="claude-sonnet-4-5",
                model_id="anthropic:claude-sonnet-4-5",
            )
        ),
    )
    session.session = SimpleNamespace(query=lambda *args, **kwargs: None)

    class FakeRunResult:
        response = SimpleNamespace(usage=SimpleNamespace(input_tokens=150_000))
        usage = RunUsage(input_tokens=300_000, output_tokens=0, requests=2)

        def all_messages(self) -> list[str]:
            return ["message"]

        def new_messages(self) -> list[ModelResponse]:
            return [
                ModelResponse(
                    parts=[TextPart(content="first")],
                    usage=RequestUsage(input_tokens=150_000, output_tokens=0),
                ),
                ModelResponse(
                    parts=[TextPart(content="second")],
                    usage=RequestUsage(input_tokens=150_000, output_tokens=0),
                ),
            ]

    class FakeStreamingHandler:
        def __init__(self, target_app: ChatApp) -> None:
            self.app = target_app

        async def execute_streaming_query(self, *args, **kwargs) -> FakeRunResult:
            _ = args, kwargs
            return FakeRunResult()

    session.streaming_handler = FakeStreamingHandler(app)

    await session._execute_query_with_cancellation("show revenue")

    viewport = "\n".join(app.render_plain_viewport())
    assert "Usage: ↑300.0k ↓0" in viewport
    assert "Ctx: 150.0k" in viewport
    assert "Cost: $0.9000" in viewport


def test_chat_app_renders_rich_output_as_ansi_inside_tui() -> None:
    terminal = FakeTerminal(columns=100, rows=12)
    app = build_chat_app(terminal=terminal, on_submit=lambda text: None)
    app.tui.start()

    table = Table(title="Results")
    table.add_column("name")
    table.add_column("total")
    table.add_row("Alice", "42")
    app.append_rich(lambda console: console.print(table))
    app.tui.flush_render()

    viewport = "\n".join(app.render_plain_viewport())
    assert "Results" in viewport
    assert "Alice" in viewport
    assert "42" in viewport
    assert app.editor.focused is True


def test_chat_app_uses_native_saber_tui_markdown() -> None:
    terminal = FakeTerminal(columns=80, rows=12)
    app = build_chat_app(terminal=terminal, on_submit=lambda text: None)
    app.tui.start()

    component = app.append_markdown("**streamed**")
    component.append_text(" markdown")
    app.tui.flush_render()

    assert isinstance(component, Markdown)
    assert app.freeze_markdown(component) is component
    assert "streamed markdown" in "\n".join(app.render_plain_viewport())


def test_chat_app_appends_theme_matched_tool_panel() -> None:
    app = build_chat_app(
        terminal=FakeTerminal(columns=80, rows=12), on_submit=lambda text: None
    )

    panel = app.append_panel()
    panel.append_markdown("**Analyzing data**\n\nFind the strongest peaks")
    panel.append_markdown("```python\nimport json\nprint(json.dumps({}))\n```")

    component = app.chat_container.children[-1]
    assert isinstance(component, Box)
    rendered = component.render(80)
    assert any("Analyzing data" in strip_ansi(line) for line in rendered)
    assert all("\x1b[48;2;" in line for line in rendered)
    assert all(visible_width(line) == 80 for line in rendered)
    assert any("\x1b[39;00m\x1b[48;2;" in line for line in rendered)


def test_chat_app_appends_native_terminal_image() -> None:
    from PIL import Image as PillowImage

    png = BytesIO()
    PillowImage.new("RGB", (8, 4), "red").save(png, format="PNG")

    app = build_chat_app(
        terminal=FakeTerminal(columns=80, rows=12), on_submit=lambda text: None
    )
    app.tui.capabilities = TerminalCapabilities("kitty")
    component = app.append_image(
        png.getvalue(),
        "image/png",
        filename="plot_1.png",
        max_width_cells=40,
        max_height_cells=10,
    )

    assert isinstance(component, TUIImage)
    assert component.options.filename == "plot_1.png"
    assert component.options.max_width_cells == 40
    assert component.options.max_height_cells == 10
    assert component in app.chat_container.children
    assert component.render(80)[0].startswith("\x1b_G")


def test_ansi_block_wraps_once_per_width(monkeypatch) -> None:
    calls = 0
    original_wrap = tui_chat.wrap_text_with_ansi

    def wrap(text, width):
        nonlocal calls
        calls += 1
        return original_wrap(text, width)

    monkeypatch.setattr(tui_chat, "wrap_text_with_ansi", wrap)
    block = tui_chat._AnsiBlock("some long text")

    first = block.render(20)
    second = block.render(20)
    third = block.render(10)

    assert first == second
    assert calls == 2
    assert third != []


def test_user_message_render_reuses_cached_theme_styles(monkeypatch) -> None:
    terminal = FakeTerminal(columns=80, rows=12)
    app = build_chat_app(terminal=terminal, on_submit=lambda text: None)

    component = app.append_assistant_message("cached answer")
    first = component.render(80)

    def fail_theme_lookup(*args, **kwargs):
        raise AssertionError("theme lookup should be cached")

    monkeypatch.setattr(tui_chat, "_theme_fg", fail_theme_lookup)

    assert component.render(80) == first


def test_streaming_handler_resolves_current_display_registry() -> None:
    terminal = FakeTerminal(columns=80, rows=12)
    app = build_chat_app(terminal=terminal, on_submit=lambda text: None)
    first_registry = {}
    second_registry = {}
    current = {"registry": first_registry}
    handler = TUIStreamingQueryHandler(
        app,
        create_console(file=StringIO(), width=80, legacy_windows=False),
        display_registry_provider=lambda: current["registry"],
    )

    assert handler._resolve_display_registry() is first_registry
    current["registry"] = second_registry
    assert handler._resolve_display_registry() is second_registry


@pytest.mark.asyncio
async def test_streaming_handler_keeps_native_markdown_on_part_end() -> None:
    terminal = FakeTerminal(columns=80, rows=12)
    app = build_chat_app(terminal=terminal, on_submit=lambda text: None)
    app.tui.start()
    handler = TUIStreamingQueryHandler(
        app,
        create_console(file=StringIO(), width=80, legacy_windows=False),
    )

    await handler.on_event(PartStartEvent(index=0, part=TextPart("**hello**")))
    app.tui.flush_render()
    component = app.chat_container.children[-1]
    await handler.on_event(
        PartEndEvent(index=0, part=TextPart("**hello**"), next_part_kind=None)
    )

    assert isinstance(component, Markdown)
    assert component in app.chat_container.children
    assert "hello" in "\n".join(app.render_plain_viewport())


@pytest.mark.asyncio
async def test_streaming_handler_writes_first_markdown_chunk_immediately() -> None:
    terminal = FakeTerminal(columns=80, rows=12)
    app = build_chat_app(terminal=terminal, on_submit=lambda text: None)
    app.tui.start()
    terminal.writes.clear()
    handler = TUIStreamingQueryHandler(
        app,
        create_console(file=StringIO(), width=80, legacy_windows=False),
    )

    await handler.on_event(PartStartEvent(index=0, part=TextPart("first chunk")))

    assert "first chunk" in strip_ansi("".join(terminal.writes))


@pytest.mark.asyncio
async def test_streaming_handler_replaces_restarted_text_part() -> None:
    app = build_chat_app(terminal=FakeTerminal(columns=80), on_submit=lambda text: None)
    handler = TUIStreamingQueryHandler(
        app,
        create_console(file=StringIO(), width=80, legacy_windows=False),
    )

    await handler.on_event(PartStartEvent(index=2, part=TextPart("obsolete")))
    obsolete = handler._stream_components[2]
    await handler.on_event(PartStartEvent(index=3, part=TextPart("sibling")))
    sibling = handler._stream_components[3]
    await handler.on_event(
        PartEndEvent(
            index=2,
            part=TextPart("obsolete"),
            next_part_kind="text",
        )
    )
    await handler.on_event(PartStartEvent(index=2, part=TextPart("replacement")))

    replacement = handler._stream_components[2]
    assert replacement is not obsolete
    assert obsolete not in app.chat_container.children
    assert replacement.text == "replacement"
    markdown_components = [
        child for child in app.chat_container.children if isinstance(child, Markdown)
    ]
    assert markdown_components == [replacement, sibling]
    assert not any(
        isinstance(left, tui_chat._AnsiBlock)
        and not left.ansi_text
        and isinstance(right, tui_chat._AnsiBlock)
        and not right.ansi_text
        for left, right in zip(
            app.chat_container.children, app.chat_container.children[1:]
        )
    )


@pytest.mark.asyncio
async def test_streaming_handler_replaces_parts_across_text_sql_and_thinking() -> None:
    app = build_chat_app(terminal=FakeTerminal(columns=80), on_submit=lambda text: None)
    handler = TUIStreamingQueryHandler(
        app,
        create_console(file=StringIO(), width=80, legacy_windows=False),
    )
    index = 4

    await handler.on_event(PartStartEvent(index=index, part=TextPart("obsolete text")))
    await handler.on_event(
        PartEndEvent(
            index=index,
            part=TextPart("obsolete text"),
            next_part_kind="tool-call",
        )
    )
    await handler.on_event(
        PartStartEvent(
            index=index,
            part=ToolCallPart(
                "execute_sql",
                {"query": "SELECT replacement"},
                tool_call_id="cross-kind",
            ),
        )
    )

    markdown_components = [
        child for child in app.chat_container.children if isinstance(child, Markdown)
    ]
    assert len(markdown_components) == 1
    assert markdown_components[0].text == "```sql\nSELECT replacement\n```"
    assert index not in handler._stream_components
    assert handler._tool_call_names == {index: "execute_sql"}
    assert handler._sql_stream_components == {index: markdown_components[0]}

    await handler.on_event(
        PartStartEvent(index=index, part=ThinkingPart("replacement reasoning"))
    )

    replacement = handler._stream_components[index]
    assert app.chat_container.children == [replacement]
    assert replacement.text == "replacement reasoning"
    assert handler._stream_kinds == {index: ThinkingPart}
    assert handler._tool_call_names == {}
    assert handler._tool_call_args == {}
    assert handler._tool_call_ids == {}
    assert handler._sql_stream_components == {}
    assert handler._sql_stream_queries == {}


@pytest.mark.asyncio
async def test_streaming_handler_routes_interleaved_text_and_thinking_by_index() -> (
    None
):
    app = build_chat_app(terminal=FakeTerminal(columns=80), on_submit=lambda text: None)
    handler = TUIStreamingQueryHandler(
        app,
        create_console(file=StringIO(), width=80, legacy_windows=False),
    )

    await handler.on_event(PartStartEvent(index=0, part=TextPart("answer")))
    await handler.on_event(PartStartEvent(index=1, part=ThinkingPart("reason")))
    await handler.on_event(
        PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=" continued"))
    )
    await handler.on_event(
        PartDeltaEvent(index=1, delta=ThinkingPartDelta(content_delta="ing"))
    )

    assert handler._stream_components[0].text == "answer continued"
    assert handler._stream_components[1].text == "reasoning"
    await handler.on_event(
        PartEndEvent(
            index=0,
            part=TextPart("answer continued"),
            next_part_kind="thinking",
        )
    )
    assert 0 in handler._stream_components
    assert 0 in handler._finished_stream_indexes
    assert 1 in handler._stream_components


@pytest.mark.asyncio
async def test_streaming_handler_renders_sql_and_results_as_native_markdown() -> None:
    terminal = FakeTerminal(columns=80, rows=20)
    app = build_chat_app(terminal=terminal, on_submit=lambda text: None)
    app.tui.start()
    handler = TUIStreamingQueryHandler(
        app,
        create_console(file=StringIO(), width=80, legacy_windows=False),
    )

    await handler.on_event(
        FunctionToolCallEvent(ToolCallPart("execute_sql", {"query": "SELECT 42"}))
    )
    await handler.on_event(
        FunctionToolResultEvent(
            ToolReturnPart(
                "execute_sql",
                '{"success": true, "results": [{"answer": 42}]}',
            )
        )
    )
    app.tui.flush_render()

    markdown_components = [
        child for child in app.chat_container.children if isinstance(child, Markdown)
    ]
    viewport = "\n".join(app.render_plain_viewport())
    assert [component.text for component in markdown_components] == [
        "```sql\nSELECT 42\n```",
        "**Results (1 rows):**\n\n| answer |\n| --- |\n| 42 |",
    ]
    assert "SELECT 42" in viewport
    assert "answer" in viewport
    assert "42" in viewport


@pytest.mark.asyncio
async def test_streaming_handler_streams_partial_sql_tool_arguments_without_duplicate() -> (
    None
):
    terminal = FakeTerminal(columns=100, rows=20)
    app = build_chat_app(terminal=terminal, on_submit=lambda text: None)
    app.tui.start()
    handler = TUIStreamingQueryHandler(
        app,
        create_console(file=StringIO(), width=100, legacy_windows=False),
    )
    tool_call_id = "sql-stream"

    await handler.on_event(
        PartStartEvent(
            index=0,
            part=ToolCallPart(
                "execute_sql",
                '{"query":"SELECT id,',
                tool_call_id="provisional-id",
            ),
        )
    )
    component = next(
        child for child in app.chat_container.children if isinstance(child, Markdown)
    )
    assert "SELECT id," in component.text
    assert "FROM legislators" not in component.text
    assert "SELECT id," in strip_ansi("".join(terminal.writes))

    await handler.on_event(
        PartDeltaEvent(
            index=0,
            delta=ToolCallPartDelta(args_delta=' name\\nFROM legislators"}'),
        )
    )
    assert "SELECT id, name\nFROM legislators" in component.text

    await handler.on_event(
        PartEndEvent(
            index=0,
            part=ToolCallPart(
                "execute_sql",
                {"query": "SELECT id, name\nFROM legislators"},
                tool_call_id=tool_call_id,
            ),
            next_part_kind=None,
        )
    )
    await handler.on_event(
        FunctionToolCallEvent(
            ToolCallPart(
                "execute_sql",
                {"query": "SELECT id, name\nFROM legislators"},
                tool_call_id=tool_call_id,
            )
        )
    )

    markdown_components = [
        child for child in app.chat_container.children if isinstance(child, Markdown)
    ]
    assert markdown_components == [component]
    assert component.text == "```sql\nSELECT id, name\nFROM legislators\n```"


@pytest.mark.asyncio
async def test_streaming_handler_streams_args_after_fragmented_tool_name() -> None:
    app = build_chat_app(terminal=FakeTerminal(columns=80), on_submit=lambda text: None)
    handler = TUIStreamingQueryHandler(
        app,
        create_console(file=StringIO(), width=80, legacy_windows=False),
    )

    await handler.on_event(
        PartStartEvent(
            index=0,
            part=ToolCallPart(
                "execute_", {"query": "SELECT 1 + 1"}, tool_call_id="sql"
            ),
        )
    )
    assert not any(isinstance(child, Markdown) for child in app.chat_container.children)

    await handler.on_event(
        PartDeltaEvent(
            index=0,
            delta=ToolCallPartDelta(tool_name_delta="sql"),
        )
    )

    component = next(
        child for child in app.chat_container.children if isinstance(child, Markdown)
    )
    assert "SELECT 1 + 1" in component.text


@pytest.mark.asyncio
async def test_streaming_handler_reconciles_interleaved_sql_calls_by_final_id() -> None:
    app = build_chat_app(
        terminal=FakeTerminal(columns=100), on_submit=lambda text: None
    )
    handler = TUIStreamingQueryHandler(
        app,
        create_console(file=StringIO(), width=100, legacy_windows=False),
    )

    await handler.on_event(
        PartStartEvent(
            index=0,
            part=ToolCallPart("execute_sql", '{"query":', tool_call_id="provisional-0"),
        )
    )
    await handler.on_event(
        PartStartEvent(
            index=1,
            part=ToolCallPart(
                "execute_sql", {"query": "SELECT 2"}, tool_call_id="provisional-1"
            ),
        )
    )
    await handler.on_event(
        PartDeltaEvent(index=0, delta=ToolCallPartDelta(args_delta='"SELECT 1"}'))
    )
    components = [handler._sql_stream_components[index] for index in (0, 1)]

    for index, query in ((0, "SELECT 1"), (1, "SELECT 2")):
        await handler.on_event(
            PartEndEvent(
                index=index,
                part=ToolCallPart(
                    "execute_sql", {"query": query}, tool_call_id=f"final-{index}"
                ),
                next_part_kind=None,
            )
        )

    for index in (1, 0):
        await handler.on_event(
            FunctionToolCallEvent(
                ToolCallPart(
                    "execute_sql",
                    {"query": f"SELECT {index + 1}"},
                    tool_call_id=f"final-{index}",
                )
            )
        )

    markdown_components = [
        child for child in app.chat_container.children if isinstance(child, Markdown)
    ]
    assert markdown_components == components
    assert app.chat_container.children == [
        components[0],
        next(
            child
            for child in app.chat_container.children
            if isinstance(child, tui_chat._AnsiBlock)
        ),
        components[1],
    ]
    assert [component.text for component in components] == [
        "```sql\nSELECT 1\n```",
        "```sql\nSELECT 2\n```",
    ]
    assert handler._tool_call_names == {}
    assert handler._sql_stream_components == {}


@pytest.mark.asyncio
async def test_streaming_handler_removes_sql_preview_rejected_by_final_call() -> None:
    app = build_chat_app(terminal=FakeTerminal(columns=80), on_submit=lambda text: None)
    handler = TUIStreamingQueryHandler(
        app,
        create_console(file=StringIO(), width=80, legacy_windows=False),
    )
    tool_call_id = "invalid-sql"

    await handler.on_event(
        PartStartEvent(
            index=0,
            part=ToolCallPart(
                "execute_sql",
                {"query": "SELECT stale_preview"},
                tool_call_id=tool_call_id,
            ),
        )
    )
    assert any(isinstance(child, Markdown) for child in app.chat_container.children)

    await handler.on_event(
        FunctionToolCallEvent(
            ToolCallPart(
                "execute_sql",
                {"query": "SELECT stale_preview", "unexpected": True},
                tool_call_id=tool_call_id,
            ),
            args_valid=False,
        )
    )

    assert not any(isinstance(child, Markdown) for child in app.chat_container.children)


@pytest.mark.asyncio
async def test_cancelled_run_removes_partial_sql_preview_and_state() -> None:
    terminal = FakeTerminal(columns=80, rows=12)
    app = build_chat_app(terminal=terminal, on_submit=lambda text: None)
    app.tui.start()
    terminal.writes.clear()
    handler = TUIStreamingQueryHandler(
        app,
        create_console(file=StringIO(), width=80, legacy_windows=False),
    )

    async def run_query(*args, event_stream_handler, **kwargs):
        async def events():
            yield PartStartEvent(
                index=0,
                part=ToolCallPart(
                    "execute_sql",
                    '{"query":"SELECT unfinished',
                    tool_call_id="cancelled-sql",
                ),
            )
            raise asyncio.CancelledError

        await event_stream_handler(SimpleNamespace(messages=[]), events())

    result = await handler.execute_streaming_query("cancel", run_query)

    assert result is None
    assert "SELECT unfinished" in strip_ansi("".join(terminal.writes))
    assert not any(isinstance(child, Markdown) for child in app.chat_container.children)
    assert handler._tool_call_names == {}
    assert handler._tool_call_args == {}
    assert handler._tool_call_ids == {}
    assert handler._sql_stream_components == {}
    assert handler._sql_stream_queries == {}
    assert not any(
        isinstance(child, tui_chat._AnsiBlock) and not child.ansi_text
        for child in app.chat_container.children
    )


@pytest.mark.asyncio
async def test_successful_run_preserves_reconciled_sql_after_state_cleanup() -> None:
    app = build_chat_app(terminal=FakeTerminal(columns=80), on_submit=lambda text: None)
    handler = TUIStreamingQueryHandler(
        app,
        create_console(file=StringIO(), width=80, legacy_windows=False),
    )
    run_result = object()
    tool_call_id = "successful-sql"

    async def run_query(*args, event_stream_handler, **kwargs):
        async def events():
            yield PartStartEvent(
                index=0,
                part=ToolCallPart(
                    "execute_sql",
                    '{"query":"SELECT preserved"',
                    tool_call_id=tool_call_id,
                ),
            )
            yield PartEndEvent(
                index=0,
                part=ToolCallPart(
                    "execute_sql",
                    {"query": "SELECT preserved"},
                    tool_call_id=tool_call_id,
                ),
                next_part_kind=None,
            )
            yield FunctionToolCallEvent(
                ToolCallPart(
                    "execute_sql",
                    {"query": "SELECT preserved"},
                    tool_call_id=tool_call_id,
                )
            )

        await event_stream_handler(SimpleNamespace(messages=[]), events())
        return run_result

    assert await handler.execute_streaming_query("success", run_query) is run_result

    markdown_components = [
        child for child in app.chat_container.children if isinstance(child, Markdown)
    ]
    assert [component.text for component in markdown_components] == [
        "```sql\nSELECT preserved\n```"
    ]
    assert handler._tool_call_names == {}
    assert handler._tool_call_args == {}
    assert handler._tool_call_ids == {}
    assert handler._sql_stream_components == {}
    assert handler._sql_stream_queries == {}


@pytest.mark.asyncio
async def test_streaming_handler_prefers_native_tool_result_tui() -> None:
    class NativeRenderer:
        def render_executing_tui(self, tui, args: dict) -> bool:
            tui.append_markdown(f"native request: {args['goal']}")
            return True

        def render_result_tui(
            self,
            tui,
            result: object,
            *,
            tool_call_id: str | None = None,
            metadata: object = None,
        ) -> bool:
            tui.append_markdown(f"native {result} ({tool_call_id}, {metadata})")
            return True

    app = build_chat_app(
        terminal=FakeTerminal(columns=80, rows=20), on_submit=lambda text: None
    )
    handler = TUIStreamingQueryHandler(
        app,
        create_console(file=StringIO(), width=80, legacy_windows=False),
        display_registry={"analyze_data": NativeRenderer()},
    )

    await handler.on_event(
        FunctionToolCallEvent(
            ToolCallPart(
                "analyze_data",
                {"goal": "Find peaks"},
                tool_call_id="native-call",
            )
        )
    )
    assert app.status.text == "Analyzing data..."

    await handler.on_event(
        FunctionToolResultEvent(
            ToolReturnPart(
                "analyze_data",
                "result",
                tool_call_id="native-call",
                metadata={"source": "test"},
            )
        )
    )

    components = [
        child for child in app.chat_container.children if isinstance(child, Markdown)
    ]
    assert [component.text for component in components] == [
        "native request: Find peaks",
        "native result (native-call, {'source': 'test'})",
    ]


@pytest.mark.asyncio
async def test_sql_result_replaces_loader_without_idle_render(monkeypatch) -> None:
    app = build_chat_app(
        terminal=FakeTerminal(columns=80, rows=20), on_submit=lambda text: None
    )
    handler = TUIStreamingQueryHandler(
        app,
        create_console(file=StringIO(), width=80, legacy_windows=False),
    )
    app.set_loading("Executing SQL...")

    monkeypatch.setattr(
        app,
        "clear_status",
        lambda: pytest.fail("result rendering must not create an idle status frame"),
    )
    await handler.on_event(
        FunctionToolResultEvent(
            ToolReturnPart(
                "execute_sql",
                '{"success": true, "results": [{"name": "Andrew Jackson"}, '
                '{"name": "Martin Van Buren"}]}',
            )
        )
    )

    assert app.status.loader is not None
    assert app.status.text == "Crunching data..."


@pytest.mark.asyncio
async def test_streaming_handler_uses_safe_fence_and_escapes_terminal_controls() -> (
    None
):
    app = build_chat_app(terminal=FakeTerminal(columns=80), on_submit=lambda text: None)
    handler = TUIStreamingQueryHandler(
        app,
        create_console(file=StringIO(), width=80, legacy_windows=False),
    )

    await handler.on_event(
        FunctionToolCallEvent(
            ToolCallPart("execute_sql", {"query": "SELECT '```'\x1b[2J"})
        )
    )
    await handler.on_event(PartStartEvent(index=0, part=TextPart("answer\x1b[2J")))

    markdown_components = [
        child for child in app.chat_container.children if isinstance(child, Markdown)
    ]
    assert markdown_components[0].text == ("````sql\nSELECT '```'\\x1B[2J\n````")
    assert markdown_components[-1].text == "answer\\x1B[2J"


@pytest.mark.asyncio
async def test_cancel_current_task_uses_token_without_task_cancellation() -> None:
    terminal = FakeTerminal(columns=80, rows=12)
    app = build_chat_app(terminal=terminal, on_submit=lambda text: None)
    app.tui.start()
    chat_console = tui_chat.ChatConsole(app)
    session = InteractiveSession.__new__(InteractiveSession)
    session.cancellation_token = asyncio.Event()

    class FakeTask:
        def __init__(self) -> None:
            self.cancel_called = False
            self._done = False

        def done(self) -> bool:
            return self._done

        def cancel(self) -> None:
            self.cancel_called = True

        def __await__(self):
            async def wait_until_token_set() -> None:
                if session.cancellation_token is not None:
                    await session.cancellation_token.wait()
                self._done = True

            return wait_until_token_set().__await__()

    task = FakeTask()
    session.current_task = task

    await session._cancel_current_task(app, chat_console)

    assert session.cancellation_token.is_set()
    assert task.done()
    assert task.cancel_called is False


@pytest.mark.asyncio
async def test_cancel_current_task_hard_cancels_when_token_does_not_finish() -> None:
    terminal = FakeTerminal(columns=80, rows=12)
    app = build_chat_app(terminal=terminal, on_submit=lambda text: None)
    app.tui.start()
    chat_console = tui_chat.ChatConsole(app)
    session = InteractiveSession.__new__(InteractiveSession)
    session.cancellation_token = asyncio.Event()
    started = asyncio.Event()

    async def blocked_query() -> None:
        started.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(blocked_query())
    await started.wait()
    session.current_task = task

    cancel_task = asyncio.create_task(session._cancel_current_task(app, chat_console))
    done, _ = await asyncio.wait({cancel_task}, timeout=0.5)

    try:
        assert cancel_task in done
        await cancel_task
        assert session.cancellation_token.is_set()
        assert task.cancelled()
    finally:
        if not cancel_task.done():
            cancel_task.cancel()
            try:
                await cancel_task
            except asyncio.CancelledError:
                pass
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


@pytest.mark.asyncio
async def test_cancel_current_task_cancels_handoff_editing() -> None:
    terminal = FakeTerminal(columns=80, rows=12)
    app = build_chat_app(terminal=terminal, on_submit=lambda text: None)
    app.tui.start()
    chat_console = tui_chat.ChatConsole(app)
    session = InteractiveSession.__new__(InteractiveSession)
    session.current_task = None
    session._handoff_mode = True
    app.editor.set_text("draft handoff prompt")
    app.set_status("Edit the handoff draft and press Enter to start a new thread.")

    await session._cancel_current_task(app, chat_console)

    assert session._handoff_mode is False
    assert app.editor.get_text() == ""
    assert "Press Ctrl+D" not in "\n".join(app.render_plain_viewport())


@pytest.mark.asyncio
async def test_interactive_session_routes_empty_submit_only_during_handoff(
    monkeypatch,
) -> None:
    terminal = FakeTerminal(columns=80, rows=12)
    session = InteractiveSession.__new__(InteractiveSession)
    session.log = type("FakeLog", (), {"info": lambda *args, **kwargs: None})()
    session.autocomplete_provider = None
    session.console = create_console(file=StringIO(), width=80, legacy_windows=False)
    session.database_name = "test"
    session._handoff_mode = False
    session.current_task = None
    session._submit_pending = False
    session._exit_finalized = False
    session.message_history = []
    session.before_prompt_loop = lambda: asyncio.sleep(0)
    session._footer_text = lambda: None
    session._load_history = lambda: []
    session.show_welcome_message = lambda app: None
    session._finalize_exit = lambda: asyncio.sleep(0)

    captured: dict[str, Callable[[], bool]] = {}

    def fake_build_chat_app(**kwargs):
        captured["should_submit_empty"] = kwargs["should_submit_empty"]
        app = build_chat_app(terminal=terminal, **kwargs)
        asyncio.get_running_loop().call_soon(app.stop)
        return app

    monkeypatch.setattr(interactive, "build_chat_app", fake_build_chat_app)

    await session.run()

    assert captured["should_submit_empty"]() is False
    session._handoff_mode = True
    assert captured["should_submit_empty"]() is True


@pytest.mark.asyncio
async def test_interactive_session_rejects_running_query_submit_without_echo(
    monkeypatch,
) -> None:
    terminal = FakeTerminal(columns=80, rows=12)
    session = InteractiveSession.__new__(InteractiveSession)
    session.log = type("FakeLog", (), {"info": lambda *args, **kwargs: None})()
    session.autocomplete_provider = None
    session.console = create_console(file=StringIO(), width=80, legacy_windows=False)
    session.database_name = "test"
    session._handoff_mode = False
    session.current_task = None
    session._submit_pending = False
    session._exit_finalized = False
    session.message_history = []
    session.before_prompt_loop = lambda: asyncio.sleep(0)
    session._footer_text = lambda: None
    session._load_history = lambda: []
    session.show_welcome_message = lambda app: None
    session._finalize_exit = lambda: asyncio.sleep(0)

    captured: dict[str, ChatApp] = {}

    def fake_build_chat_app(**kwargs):
        app = build_chat_app(terminal=terminal, **kwargs)
        captured["app"] = app
        asyncio.get_running_loop().call_soon(app.stop)
        return app

    monkeypatch.setattr(interactive, "build_chat_app", fake_build_chat_app)

    await session.run()
    app = captured["app"]

    class RunningTask:
        def done(self) -> bool:
            return False

    session.current_task = RunningTask()
    app.editor.set_text("still running")

    app.submit("still running")

    assert app.editor.get_text() == "still running"
    assert app.editor.history == []
    assert all("still running" not in line for line in app.chat_container.render(80))


@pytest.mark.asyncio
async def test_streaming_handler_interrupts_with_regular_exception() -> None:
    terminal = FakeTerminal(columns=80, rows=12)
    app = build_chat_app(terminal=terminal, on_submit=lambda text: None)
    app.tui.start()
    handler = TUIStreamingQueryHandler(
        app,
        create_console(file=StringIO(), width=80, legacy_windows=False),
    )
    cancellation_token = asyncio.Event()
    saw_cancelled_error = False

    async def run_query(*args, **kwargs):
        _ = args
        event_stream_handler = kwargs["event_stream_handler"]

        async def events():
            cancellation_token.set()
            yield PartStartEvent(index=0, part=TextPart("first"))

        try:
            await event_stream_handler(None, events())
        except asyncio.CancelledError:
            nonlocal saw_cancelled_error
            saw_cancelled_error = True
            raise

    result = await handler.execute_streaming_query(
        "interrupt",
        run_query,
        cancellation_token=cancellation_token,
    )

    assert result is None
    assert saw_cancelled_error is False


@pytest.mark.asyncio
async def test_streaming_handler_token_does_not_preempt_blocked_run_query() -> None:
    terminal = FakeTerminal(columns=80, rows=12)
    app = build_chat_app(terminal=terminal, on_submit=lambda text: None)
    app.tui.start()
    handler = TUIStreamingQueryHandler(
        app,
        create_console(file=StringIO(), width=80, legacy_windows=False),
    )
    started = asyncio.Event()
    release = asyncio.Event()
    run_cancelled = asyncio.Event()
    cancellation_token = asyncio.Event()

    async def run_query(*args, **kwargs):
        _ = args, kwargs
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            run_cancelled.set()
            raise

    task = asyncio.create_task(
        handler.execute_streaming_query(
            "slow query",
            run_query,
            cancellation_token=cancellation_token,
        )
    )
    await started.wait()

    cancellation_token.set()
    done, _ = await asyncio.wait({task}, timeout=0.05)
    release.set()

    try:
        assert task not in done
        assert await task is None
        assert run_cancelled.is_set() is False
    finally:
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


@pytest.mark.asyncio
async def test_streaming_handler_cancels_run_query_when_task_is_cancelled() -> None:
    terminal = FakeTerminal(columns=80, rows=12)
    app = build_chat_app(terminal=terminal, on_submit=lambda text: None)
    app.tui.start()
    handler = TUIStreamingQueryHandler(
        app,
        create_console(file=StringIO(), width=80, legacy_windows=False),
    )
    started = asyncio.Event()
    run_cancelled = asyncio.Event()

    async def run_query(*args, **kwargs):
        _ = args, kwargs
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            run_cancelled.set()
            raise

    task = asyncio.create_task(handler.execute_streaming_query("slow query", run_query))
    await started.wait()

    task.cancel()

    assert await task is None
    assert run_cancelled.is_set()


@pytest.mark.asyncio
async def test_streaming_handler_runs_query_in_current_task() -> None:
    terminal = FakeTerminal(columns=80, rows=12)
    app = build_chat_app(terminal=terminal, on_submit=lambda text: None)
    app.tui.start()
    handler = TUIStreamingQueryHandler(
        app,
        create_console(file=StringIO(), width=80, legacy_windows=False),
    )
    execute_task = asyncio.current_task()
    run_query_task: asyncio.Task | None = None

    async def run_query(*args, **kwargs):
        _ = args, kwargs
        nonlocal run_query_task
        run_query_task = asyncio.current_task()
        return None

    await handler.execute_streaming_query("task check", run_query)

    assert run_query_task is execute_task


def test_ctrl_d_on_empty_editor_requests_exit() -> None:
    terminal = FakeTerminal()
    exited = False

    def on_exit() -> None:
        nonlocal exited
        exited = True

    app = build_chat_app(
        terminal=terminal, on_submit=lambda text: None, on_exit=on_exit
    )
    app.tui.start()

    terminal.send_input("\x04")

    assert exited is True
    assert terminal.stopped is True
