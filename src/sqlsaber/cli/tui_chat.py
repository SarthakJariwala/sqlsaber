"""Persistent saber-tui chat surface for interactive SQLsaber sessions."""

from __future__ import annotations

import io
import platform
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import saber_tui.utils as tui_utils
from pygments import highlight
from pygments.formatters.terminal256 import TerminalTrueColorFormatter
from pygments.lexers import get_lexer_by_name
from pygments.lexers.special import TextLexer
from pygments.styles import get_style_by_name
from pygments.util import ClassNotFound
from rich.console import Console
from saber_tui import (
    PosixProcessTerminal,
    TUI,
    Terminal,
    WindowsProcessTerminal,
    matches_key,
)
from saber_tui.components import (
    CancellableLoader,
    DefaultTextStyle,
    Editor,
    EditorTheme,
    Markdown,
    MarkdownTheme,
    SettingItem,
    SettingsList,
    SettingsListOptions,
    SettingsListTheme,
)
from saber_tui.components.select_list import SelectListTheme
from saber_tui.utils import (
    apply_background_to_line,
    strip_ansi,
    truncate_to_width,
    wrap_text_with_ansi,
)

from sqlsaber.config.settings import ThinkingLevel
from sqlsaber.theme.manager import get_theme_manager

type ColorSystem = Literal["auto", "standard", "256", "truecolor", "windows"]
type SubmitHandler = Callable[[str], bool | None]
type ThinkingChangeHandler = Callable[[bool, ThinkingLevel | None], None]


def _fg(r: int, g: int, b: int) -> Callable[[str], str]:
    code = f"\x1b[38;2;{r};{g};{b}m"
    return lambda text: f"{code}{text}\x1b[39m"


def _bg(r: int, g: int, b: int) -> Callable[[str], str]:
    code = f"\x1b[48;2;{r};{g};{b}m"
    return lambda text: f"{code}{text}\x1b[49m"


def _bold(text: str) -> str:
    return f"\x1b[1m{text}\x1b[22m"


def _italic(text: str) -> str:
    return f"\x1b[3m{text}\x1b[23m"


def _strikethrough(text: str) -> str:
    return f"\x1b[9m{text}\x1b[29m"


def _underline(text: str) -> str:
    return f"\x1b[4m{text}\x1b[24m"


INFO_FALLBACK = (125, 211, 252)
SUCCESS_FALLBACK = (134, 239, 172)
WARNING_FALLBACK = (251, 191, 36)
MUTED_FALLBACK = (148, 163, 184)
USER_BG_FALLBACK = (30, 41, 59)
MIN_TUI_WIDTH_CACHE_SIZE = 65_536
SHIFT_ENTER_SEQUENCE = "\x1b[13;2u"
SHIFT_ENTER_FALLBACK_SEQUENCES = {"\n", "\x1b\r"}
THINKING_MODE_SETTING_ID = "thinking_mode"
THINKING_MODE_VALUES = ["off", *(level.value for level in ThinkingLevel)]
ACTION_VALUE = ""
COMMAND_PALETTE_MAX_WIDTH = 60
DANGEROUS_MODE_FOOTER_LABEL = "Dangerous mode"


def _default_process_terminal() -> Terminal:
    if platform.system() == "Windows":
        return WindowsProcessTerminal()
    return PosixProcessTerminal()


def _ensure_tui_width_cache_capacity() -> None:
    current_size = getattr(tui_utils, "_WIDTH_CACHE_SIZE", 0)
    if current_size < MIN_TUI_WIDTH_CACHE_SIZE:
        setattr(tui_utils, "_WIDTH_CACHE_SIZE", MIN_TUI_WIDTH_CACHE_SIZE)


def _hex_to_rgb(color: str | None) -> tuple[int, int, int] | None:
    if not color:
        return None
    normalized = color.strip()
    if normalized.startswith("#"):
        normalized = normalized[1:]
    if len(normalized) == 3:
        normalized = "".join(ch * 2 for ch in normalized)
    if len(normalized) != 6:
        return None
    try:
        return (
            int(normalized[0:2], 16),
            int(normalized[2:4], 16),
            int(normalized[4:6], 16),
        )
    except ValueError:
        return None


def _blend(
    base: tuple[int, int, int], accent: tuple[int, int, int], ratio: float
) -> tuple[int, int, int]:
    return (
        round(base[0] * (1 - ratio) + accent[0] * ratio),
        round(base[1] * (1 - ratio) + accent[1] * ratio),
        round(base[2] * (1 - ratio) + accent[2] * ratio),
    )


def _theme_fg_rgb(role: str, fallback: tuple[int, int, int]) -> tuple[int, int, int]:
    tm = get_theme_manager()
    console = Console(theme=tm.rich_theme, color_system="truecolor")
    try:
        style = console.get_style(role)
    except Exception:
        style = None
    if style is not None and style.color is not None:
        color = style.color.get_truecolor()
        return color.red, color.green, color.blue
    return fallback


def _theme_fg(role: str, fallback: tuple[int, int, int]) -> Callable[[str], str]:
    return _fg(*_theme_fg_rgb(role, fallback))


def _theme_user_bg() -> Callable[[str], str]:
    tm = get_theme_manager()
    try:
        style = get_style_by_name(tm.pygments_style_name)
    except ClassNotFound:
        return _bg(*USER_BG_FALLBACK)

    highlight = _hex_to_rgb(getattr(style, "highlight_color", None))
    if highlight is not None:
        return _bg(*highlight)

    background = _hex_to_rgb(getattr(style, "background_color", None))
    if background is None:
        return _bg(*USER_BG_FALLBACK)

    accent = _theme_fg_rgb("panel.border.user", INFO_FALLBACK)
    return _bg(*_blend(background, accent, 0.16))


@dataclass(frozen=True)
class _TUITheme:
    user_fg: Callable[[str], str]
    assistant_fg: Callable[[str], str]
    system_fg: Callable[[str], str]
    muted_fg: Callable[[str], str]
    spinner_fg: Callable[[str], str]
    status_fg: Callable[[str], str]
    user_bg: Callable[[str], str]
    markdown: MarkdownTheme


def _build_tui_theme() -> _TUITheme:
    primary = _theme_fg("primary", INFO_FALLBACK)
    accent = _theme_fg("accent", SUCCESS_FALLBACK)
    info = _theme_fg("info", INFO_FALLBACK)
    warning = _theme_fg("warning", WARNING_FALLBACK)
    muted = _theme_fg("muted", MUTED_FALLBACK)
    tm = get_theme_manager()
    formatter = TerminalTrueColorFormatter(style=tm.pygments_style_name)

    def highlight_code(code: str, language: str | None) -> list[str]:
        try:
            lexer = get_lexer_by_name(language) if language else TextLexer()
        except ClassNotFound:
            lexer = TextLexer()
        rendered = highlight(code, lexer, formatter).rstrip("\n")
        return rendered.split("\n") if rendered else []

    return _TUITheme(
        user_fg=_theme_fg("panel.border.user", INFO_FALLBACK),
        assistant_fg=_theme_fg("panel.border.assistant", SUCCESS_FALLBACK),
        system_fg=_theme_fg("warning", WARNING_FALLBACK),
        muted_fg=muted,
        spinner_fg=_theme_fg("spinner", WARNING_FALLBACK),
        status_fg=_theme_fg("status", WARNING_FALLBACK),
        user_bg=_theme_user_bg(),
        markdown=MarkdownTheme(
            heading=primary,
            link=info,
            link_url=muted,
            code=warning,
            code_block_border=muted,
            quote=muted,
            quote_border=muted,
            hr=muted,
            list_bullet=accent,
            bold=_bold,
            italic=_italic,
            strikethrough=_strikethrough,
            underline=_underline,
            highlight_code=highlight_code,
        ),
    )


def _pad_to_width(text: str, width: int) -> str:
    return truncate_to_width(text, width, "", pad=True)


class _AnsiBlock:
    def __init__(self, ansi_text: str) -> None:
        self.ansi_text = ansi_text.rstrip("\n")
        self._cache_width: int | None = None
        self._cache_lines: list[str] | None = None

    def render(self, width: int) -> list[str]:
        if self._cache_width == width and self._cache_lines is not None:
            return list(self._cache_lines)

        if not self.ansi_text:
            lines = [""]
            self._cache_width = width
            self._cache_lines = lines
            return list(lines)

        lines: list[str] = []
        for line in self.ansi_text.splitlines():
            lines.extend(wrap_text_with_ansi(line, max(1, width)))
        lines = lines or [""]
        self._cache_width = width
        self._cache_lines = lines
        return list(lines)

    def invalidate(self) -> None:
        self._cache_width = None
        self._cache_lines = None


@dataclass
class _MessageComponent:
    role: str
    text: str
    theme: _TUITheme
    _cache_width: int | None = field(default=None, init=False)
    _cache_lines: list[str] | None = field(default=None, init=False)

    def render(self, width: int) -> list[str]:
        if self._cache_width == width and self._cache_lines is not None:
            return list(self._cache_lines)

        if self.role == "user":
            lines = self._render_user(width)
            self._cache_width = width
            self._cache_lines = lines
            return list(lines)

        label, style = self._role_style()
        label_plain = f"  {label}: "
        prefix_width = len(label_plain)
        prefix = f"  {_bold(style(label))}: "
        indent = " " * prefix_width
        content_width = max(1, width - prefix_width - 2)
        wrapped = wrap_text_with_ansi(self.text, content_width) or [""]
        lines: list[str] = []
        for index, line in enumerate(wrapped):
            styled = style(line)
            lines.append(prefix + styled if index == 0 else indent + styled)
        self._cache_width = width
        self._cache_lines = lines
        return list(lines)

    def _render_user(self, width: int) -> list[str]:
        if width <= 0:
            return [""]

        content_width = max(1, width)
        wrapped: list[str] = []
        for logical_line in self.text.split("\n"):
            wrapped.extend(wrap_text_with_ansi(logical_line, content_width) or [""])

        return [
            self.theme.user_bg(" " * width),
            *[
                apply_background_to_line(
                    self.theme.user_fg(line), width, self.theme.user_bg
                )
                for line in wrapped
            ],
            self.theme.user_bg(" " * width),
        ]

    def invalidate(self) -> None:
        self._cache_width = None
        self._cache_lines = None

    def _role_style(self) -> tuple[str, Callable[[str], str]]:
        if self.role == "user":
            return "You", self.theme.user_fg
        if self.role == "assistant":
            return "SQLsaber", self.theme.assistant_fg
        return "System", self.theme.system_fg


class _StatusComponent:
    def __init__(
        self,
        tui: TUI,
        theme: _TUITheme,
        on_cancel: Callable[[], None] | None = None,
    ) -> None:
        self.tui = tui
        self.theme = theme
        self.on_cancel = on_cancel
        self.text: str | None = None
        self.loader: CancellableLoader | None = None

    def set_text(self, text: str | None, *, loading: bool = False) -> None:
        self._dispose_loader()
        self.text = text
        if text is not None and loading:
            self.loader = CancellableLoader(
                self.tui,
                spinner_style=self.theme.spinner_fg,
                text_style=self.theme.status_fg,
                text=text,
            )
            self.loader.on_cancel = self.on_cancel

    def render(self, width: int) -> list[str]:
        loader = self.loader
        if loader is not None:
            return loader.render(width)
        if not self.text:
            return [" " * width]
        return [self.theme.status_fg(_pad_to_width(f"  {self.text}", width))]

    def invalidate(self) -> None:
        return None

    def cancel_loading(self) -> bool:
        loader = self.loader
        if loader is None:
            return False
        loader.handle_input("\x03")
        return True

    def dispose(self) -> None:
        self._dispose_loader()

    def _dispose_loader(self) -> None:
        loader = self.loader
        self.loader = None
        if loader is not None:
            loader.dispose()


class _FooterComponent:
    def __init__(self, text: str | None, theme: _TUITheme) -> None:
        self.text = text
        self.theme = theme

    def set_text(self, text: str | None) -> None:
        self.text = text

    def render(self, width: int) -> list[str]:
        if not self.text:
            return []
        line = _pad_to_width(f" {self.text}", width)
        return [self._style_line(line)]

    def _style_line(self, line: str) -> str:
        if DANGEROUS_MODE_FOOTER_LABEL not in line:
            return self.theme.muted_fg(line)

        before, _, after = line.partition(DANGEROUS_MODE_FOOTER_LABEL)
        return (
            self.theme.muted_fg(before)
            + _bold(self.theme.system_fg(DANGEROUS_MODE_FOOTER_LABEL))
            + self.theme.muted_fg(after)
        )

    def invalidate(self) -> None:
        return None


class _CommandPaletteComponent:
    def __init__(self, settings: SettingsList, theme: _TUITheme) -> None:
        self.settings = settings
        self.theme = theme

    def handle_input(self, data: str) -> None:
        self.settings.handle_input(data)

    def render(self, width: int) -> list[str]:
        if width <= 0:
            return [""]

        panel_width = max(1, min(COMMAND_PALETTE_MAX_WIDTH, width))
        content_width = max(1, panel_width - 2)
        border = self.theme.muted_fg("─" * width)
        lines = [border]
        for line in self.settings.render(content_width):
            if "Enter/Space to change" in strip_ansi(line):
                line = self.theme.muted_fg(
                    "  Type to search - Enter/Space to run or change - Esc to cancel"
                )
            lines.append(f" {line}")
        lines.append(border)
        return [_pad_to_width(line, width) for line in lines]

    def invalidate(self) -> None:
        self.settings.invalidate()


def _build_settings_list_theme(theme: _TUITheme) -> SettingsListTheme:
    def label(text: str, selected: bool) -> str:
        styled = theme.assistant_fg(text)
        return _bold(styled) if selected else styled

    def value(text: str, selected: bool) -> str:
        styled = theme.assistant_fg(text) if selected else theme.status_fg(text)
        return _bold(styled) if selected else styled

    return SettingsListTheme(
        label=label,
        value=value,
        description=theme.muted_fg,
        cursor=theme.status_fg("-> "),
        hint=theme.muted_fg,
    )


class RichCapture:
    """Render Rich output to ANSI text for inclusion in saber-tui components."""

    def __init__(self, base_console: Console | None = None) -> None:
        self._base_console = base_console

    def capture(self, render: Callable[[Console], None], width: int) -> str:
        buffer = io.StringIO()
        tm = get_theme_manager()
        console = Console(
            file=buffer,
            force_terminal=True,
            color_system=self._color_system(),
            width=max(20, width),
            theme=tm.rich_theme,
            legacy_windows=False,
        )
        render(console)
        return buffer.getvalue()

    def _color_system(self) -> ColorSystem | None:
        if self._base_console is None:
            return "truecolor"
        color_system = self._base_console.color_system
        if color_system in {"auto", "standard", "256", "truecolor", "windows"}:
            return cast(ColorSystem, color_system)
        return "truecolor"


class ChatConsole(Console):
    """Rich Console facade that appends printed output to the chat app."""

    def __init__(self, app: ChatApp) -> None:
        self._app = app
        super().__init__(
            file=io.StringIO(),
            force_terminal=True,
            color_system="truecolor",
            theme=get_theme_manager().rich_theme,
        )

    def print(self, *objects: Any, **kwargs: Any) -> None:
        self._app.append_rich(lambda console: console.print(*objects, **kwargs))

    def print_json(self, *args: Any, **kwargs: Any) -> None:
        self._app.append_rich(lambda console: console.print_json(*args, **kwargs))


class ChatApp:
    """Small persistent chat shell built on saber-tui."""

    def __init__(
        self,
        *,
        tui: TUI,
        editor: Editor,
        chat_container,
        status: _StatusComponent,
        footer: _FooterComponent,
        theme: _TUITheme,
        rich_capture: RichCapture,
        on_submit: SubmitHandler,
        on_exit: Callable[[], None] | None = None,
        on_cancel: Callable[[], None] | None = None,
        should_submit_empty: Callable[[], bool] | None = None,
        on_open_command_palette: Callable[[ChatApp], None] | None = None,
    ) -> None:
        self.tui = tui
        self.editor = editor
        self.chat_container = chat_container
        self.status = status
        self.footer = footer
        self.theme = theme
        self.rich_capture = rich_capture
        self.on_submit = on_submit
        self.on_exit = on_exit
        self.on_cancel = on_cancel
        self.should_submit_empty = should_submit_empty
        self.on_open_command_palette = on_open_command_palette
        self._command_palette_component: _CommandPaletteComponent | None = None

    def submit(self, text: str) -> None:
        text = text.strip()
        if text == "/" and self.open_command_palette():
            self.editor.set_text("")
            return
        if not text and not self._should_submit_empty():
            self.editor.set_text("")
            return
        accepted = self.on_submit(text)
        if accepted is False:
            self.tui.set_focus(self.editor)
            self.tui.request_render()
            return
        if text and not text.startswith("/"):
            self.append_user_message(text)
            self.editor.add_to_history(text)
        self.editor.set_text("")
        self.tui.set_focus(self.editor)
        self.tui.request_render()

    def _should_submit_empty(self) -> bool:
        if self.should_submit_empty is None:
            return False
        return self.should_submit_empty()

    def append_user_message(self, text: str) -> None:
        self._append_message("user", text)

    def append_assistant_message(self, text: str) -> _MessageComponent:
        return self._append_message("assistant", text)

    def append_system_message(self, text: str) -> None:
        self._append_message("system", text)

    def append_rich(self, render: Callable[[Console], None]) -> None:
        ansi = self.rich_capture.capture(render, self.tui.terminal.columns)
        self._append_component(_AnsiBlock(ansi))

    def append_markdown(self, text: str = "", *, muted: bool = False) -> Markdown:
        component = self._create_markdown(text, muted=muted)
        self._append_component(component)
        return component

    def insert_markdown_before(
        self, before: Markdown, text: str = "", *, muted: bool = False
    ) -> Markdown:
        component = self._create_markdown(text, muted=muted)
        children = self.chat_container.children
        try:
            index = children.index(before)
        except ValueError:
            self._append_component(component)
            return component
        children.insert(index, component)
        children.insert(index + 1, _AnsiBlock(""))
        self.tui.request_render()
        return component

    def replace_markdown(
        self, previous: Markdown, text: str = "", *, muted: bool = False
    ) -> Markdown:
        component = self._create_markdown(text, muted=muted)
        children = self.chat_container.children
        try:
            index = children.index(previous)
        except ValueError:
            self._append_component(component)
            return component
        children[index] = component
        self.tui.request_render()
        return component

    def _create_markdown(self, text: str, *, muted: bool) -> Markdown:
        default_style = DefaultTextStyle(color=self.theme.muted_fg) if muted else None
        return Markdown(
            text,
            theme=self.theme.markdown,
            default_text_style=default_style,
        )

    def freeze_markdown(self, component: Markdown) -> Markdown:
        """Finalize a stream; saber-tui's Markdown owns its render cache."""
        return component

    def remove_markdown(self, component: Markdown) -> None:
        """Remove a speculative Markdown component and its leading spacer."""
        children = self.chat_container.children
        try:
            index = children.index(component)
        except ValueError:
            return
        del children[index]
        if (
            index > 0
            and isinstance(children[index - 1], _AnsiBlock)
            and not children[index - 1].ansi_text
        ):
            del children[index - 1]
        elif (
            index == 0
            and children
            and isinstance(children[0], _AnsiBlock)
            and not children[0].ansi_text
        ):
            del children[0]
        self.tui.request_render()

    def set_status(self, text: str | None) -> None:
        self.status.set_text(text)
        self.tui.request_render()

    def set_loading(self, text: str) -> None:
        self.status.set_text(text, loading=True)
        self.tui.request_render()

    def clear_status(self) -> None:
        self.set_status(None)

    def set_footer(self, text: str | None) -> None:
        self.footer.set_text(text)
        self.tui.request_render()

    def open_command_palette(self) -> bool:
        if self.on_open_command_palette is None:
            return False
        self.on_open_command_palette(self)
        return True

    def is_command_palette_open(self) -> bool:
        return self._command_palette_component is not None

    def show_command_palette(
        self,
        *,
        thinking_enabled: bool,
        thinking_level: ThinkingLevel,
        on_thinking_change: ThinkingChangeHandler,
        model_name: str | None = None,
        database_name: str | None = None,
    ) -> None:
        """Open the command palette."""
        self.close_command_palette()

        current_mode = thinking_level.value if thinking_enabled else "off"

        def on_change(setting_id: str, value: str) -> None:
            if setting_id == THINKING_MODE_SETTING_ID:
                if value == "off":
                    on_thinking_change(False, None)
                    self.set_status("Thinking disabled")
                    return

                level = ThinkingLevel(value)
                on_thinking_change(True, level)
                self.set_status(f"Thinking enabled ({level.value})")
                return

            self._run_command_palette_action(setting_id)

        settings_items = [
            SettingItem(
                THINKING_MODE_SETTING_ID,
                "Thinking mode",
                current_mode,
                values=list(THINKING_MODE_VALUES),
                description="Controls provider reasoning for this interactive session.",
            ),
            SettingItem(
                "handoff",
                "Handoff thread",
                ACTION_VALUE,
                values=[ACTION_VALUE],
                description="Draft a prompt for a new thread with current context.",
            ),
            SettingItem(
                "clear",
                "Clear conversation",
                ACTION_VALUE,
                values=[ACTION_VALUE],
                description="Clear visible conversation and current thread history.",
            ),
            SettingItem(
                "exit",
                "Exit",
                ACTION_VALUE,
                values=[ACTION_VALUE],
                description="End this interactive session.",
            ),
        ]
        settings = SettingsList(
            settings_items,
            max_visible=8,
            theme=_build_settings_list_theme(self.theme),
            on_change=on_change,
            on_cancel=self.close_command_palette,
            options=SettingsListOptions(enable_search=True),
        )
        if settings.search_input is not None:
            settings.search_input.focused = True

        component = _CommandPaletteComponent(settings, self.theme)
        self._command_palette_component = component
        self._replace_editor(component)
        self.tui.set_focus(component)
        self.tui.request_render()

    def _run_command_palette_action(self, action_id: str) -> None:
        if action_id == "handoff":
            self.close_command_palette()
            self.editor.set_text("/handoff ")
            self.set_status("Type the handoff goal and press Enter.")
            self.tui.set_focus(self.editor)
            self.tui.request_render()
            return

        if action_id == "clear":
            self.close_command_palette()
            self.submit("/clear")
            return

        if action_id == "exit":
            self.close_command_palette()
            self.submit("/exit")
            return

    def close_command_palette(self) -> None:
        component = self._command_palette_component
        self._command_palette_component = None
        if component is not None:
            self._replace_command_palette_with_editor(component)
        self.tui.set_focus(self.editor)
        self.tui.request_render()

    def _replace_editor(self, component) -> None:
        try:
            index = self.tui.children.index(self.editor)
        except ValueError:
            return
        self.tui.children[index] = component

    def _replace_command_palette_with_editor(
        self, component: _CommandPaletteComponent
    ) -> None:
        try:
            index = self.tui.children.index(component)
        except ValueError:
            return
        self.tui.children[index] = self.editor

    def stop(self) -> None:
        if not self.tui.stopped:
            self.status.dispose()
            self.tui.flush_render()
            self.tui.stop()
        if self.on_exit is not None:
            self.on_exit()

    def render_plain_viewport(self) -> list[str]:
        width = self.tui.terminal.columns
        height = self.tui.terminal.rows
        lines, _ = self.tui._prepare_lines(width, height)
        return [strip_ansi(line) for line in lines[-height:]]

    def _append_message(self, role: str, text: str) -> _MessageComponent:
        component = _MessageComponent(role, text, self.theme)
        self._append_component(component)
        return component

    def _append_component(self, component) -> None:
        if self.chat_container.children:
            self.chat_container.add_child(_AnsiBlock(""))
        self.chat_container.add_child(component)
        self.tui.request_render()


def build_chat_app(
    *,
    terminal: Terminal | None = None,
    on_submit: SubmitHandler,
    on_exit: Callable[[], None] | None = None,
    on_cancel: Callable[[], None] | None = None,
    should_submit_empty: Callable[[], bool] | None = None,
    autocomplete_provider=None,
    console: Console | None = None,
    footer_text: str | None = None,
    on_open_command_palette: Callable[[ChatApp], None] | None = None,
) -> ChatApp:
    """Build the persistent interactive chat app."""
    from saber_tui import Container

    _ensure_tui_width_cache_capacity()

    term = terminal if terminal is not None else _default_process_terminal()
    tui = TUI(term)
    tui.set_show_hardware_cursor(True)
    tui.set_clear_on_shrink(False)

    theme = _build_tui_theme()
    chat_container = Container()
    status = _StatusComponent(tui, theme, on_cancel)
    footer = _FooterComponent(footer_text, theme)
    editor = Editor(
        tui,
        theme=EditorTheme(
            border_color=theme.muted_fg,
            select_list=SelectListTheme(),
        ),
    )
    if autocomplete_provider is not None:
        editor.set_autocomplete_provider(autocomplete_provider)

    app = ChatApp(
        tui=tui,
        editor=editor,
        chat_container=chat_container,
        status=status,
        footer=footer,
        theme=theme,
        rich_capture=RichCapture(console),
        on_submit=on_submit,
        on_exit=on_exit,
        on_cancel=on_cancel,
        should_submit_empty=should_submit_empty,
        on_open_command_palette=on_open_command_palette,
    )
    editor.on_submit = app.submit

    def global_listener(data: str):
        if app.is_command_palette_open() and (
            matches_key(data, "escape") or matches_key(data, "ctrl+c")
        ):
            app.close_command_palette()
            return {"consume": True}
        if (
            data == "/"
            and app.tui.focused_component is editor
            and not editor.get_text().strip()
            and app.open_command_palette()
        ):
            return {"consume": True}
        if data in SHIFT_ENTER_FALLBACK_SEQUENCES:
            return {"data": SHIFT_ENTER_SEQUENCE}
        if matches_key(data, "ctrl+d") and not editor.get_text().strip():
            if app.status.cancel_loading():
                app.tui.set_focus(app.editor)
                return {"consume": True}
            app.stop()
            return {"consume": True}
        if matches_key(data, "ctrl+c") and app.status.cancel_loading():
            app.tui.set_focus(app.editor)
            return {"consume": True}
        if matches_key(data, "ctrl+c") and app.on_cancel is not None:
            app.on_cancel()
            return {"consume": True}
        return None

    tui.add_child(chat_container)
    tui.add_child(status)
    tui.add_child(editor)
    tui.add_child(footer)
    tui.set_focus(editor)
    tui.add_input_listener(global_listener)
    return app
