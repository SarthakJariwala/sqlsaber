"""PromptForm widget plus overlay and transient-TUI hosts.

This module may import saber-tui Input, SelectList, and TUI. It must not be
imported from ``cli/commands.py`` at module load.
"""

from __future__ import annotations

import asyncio
import getpass
import platform
from collections.abc import Callable
from typing import Any, cast

from saber_tui import PosixProcessTerminal, TUI, WindowsProcessTerminal
from saber_tui.components import Input
from saber_tui.components.select_list import SelectItem, SelectList, SelectListTheme
from saber_tui.fuzzy import fuzzy_filter

from sqlsaber.render.surface import (
    Ask,
    AskChoice,
    AskConfirm,
    AskPath,
    AskSecret,
    AskText,
    Choice,
)
from sqlsaber.theme.styles import Styles, bold


def _default_process_terminal() -> Any:
    if platform.system() == "Windows":
        return WindowsProcessTerminal()
    return PosixProcessTerminal()


class PromptForm:
    """Message line plus Input or filtered SelectList plus a hint line.

    Resolving from saber-tui's stdin reader thread is the host's problem:
    the widget calls ``on_done`` / ``on_cancel``.
    """

    def __init__(
        self,
        prompt: Ask[Any],
        styles: Styles,
        *,
        on_done: Callable[[object], None],
        on_cancel: Callable[[], None],
    ) -> None:
        self.focused = False
        self._prompt = prompt
        self._styles = styles
        self._on_done = on_done
        self._on_cancel = on_cancel
        self._error: str | None = None
        self._filter = ""
        self._choices: list[Choice[Any]] = []
        self._input: Input | None = None
        self._select: SelectList | None = None
        self._build()

    def render(self, width: int) -> list[str]:
        """Render the form.

        Args:
            width: Terminal width in cells.

        Returns:
            One string per row.
        """
        lines: list[str] = []
        message = self._styles.assistant_fg(self._prompt.message)
        lines.append(message)
        if self._error:
            lines.append(self._styles.role("error")(self._error))
        if self._filter and self._select is not None:
            lines.append(self._styles.muted_fg(f"  filter: {self._filter}"))
        if self._input is not None:
            lines.extend(self._input.render(width))
        if self._select is not None:
            lines.extend(self._select.render(width))
        hint = "enter confirm · esc cancel"
        if isinstance(self._prompt, AskChoice) and self._prompt.searchable:
            hint = "type to filter · " + hint
        lines.append(self._styles.muted_fg(hint))
        return lines or [""]

    def handle_input(self, data: str) -> None:
        """Dispatch a key sequence to the active child widget.

        Args:
            data: Raw key data from saber-tui.
        """
        if self._select is not None and isinstance(self._prompt, AskChoice):
            if self._prompt.searchable and _is_filter_char(data):
                if data in {"\x7f", "\x08"}:
                    self._filter = self._filter[:-1]
                elif data.isprintable() and len(data) == 1:
                    self._filter += data
                self._apply_filter()
                return
        if self._input is not None:
            self._input.handle_input(data)
            return
        if self._select is not None:
            self._select.handle_input(data)

    def invalidate(self) -> None:
        """Drop child caches."""
        if self._input is not None:
            self._input.invalidate()
        if self._select is not None:
            self._select.invalidate()

    def _build(self) -> None:
        prompt = self._prompt
        if isinstance(prompt, AskText | AskPath):
            widget = Input()
            widget.focused = True
            widget.set_value(prompt.default)
            widget.on_submit = self._submit_text
            widget.on_escape = self._on_cancel
            self._input = widget
            return
        if isinstance(prompt, AskConfirm):
            self._choices = [
                Choice("Yes", True),
                Choice("No", False),
            ]
            self._build_select(default=prompt.default, searchable=False)
            return
        if isinstance(prompt, AskChoice):
            self._choices = list(prompt.choices)
            self._build_select(default=prompt.default, searchable=prompt.searchable)
            return
        raise TypeError(f"unsupported prompt type: {type(prompt).__name__}")

    def _build_select(self, *, default: object, searchable: bool) -> None:
        items = [
            SelectItem(
                value=str(index), label=choice.label, description=choice.description
            )
            for index, choice in enumerate(self._choices)
        ]
        selected = 0
        for index, choice in enumerate(self._choices):
            if choice.value == default:
                selected = index
                break
        theme = SelectListTheme(
            selected_prefix=lambda text: self._styles.assistant_fg(text),
            selected_text=lambda text: bold(self._styles.assistant_fg(text)),
            description=self._styles.muted_fg,
            scroll_info=self._styles.muted_fg,
            no_match=self._styles.muted_fg,
        )
        select = SelectList(items, max_visible=8, theme=theme)
        select.set_selected_index(selected)
        select.on_select = self._submit_select
        select.on_cancel = self._on_cancel
        self._select = select
        del searchable

    def _apply_filter(self) -> None:
        if self._select is None:
            return
        if not self._filter:
            self._select.filtered_items = list(self._select.items)
            self._select.selected_index = 0
            return
        labels = [item.label for item in self._select.items]
        matched = {match.value for match in fuzzy_filter(self._filter, labels)}
        self._select.filtered_items = [
            item for item in self._select.items if item.label in matched
        ]
        self._select.selected_index = 0

    def _submit_text(self, value: str) -> None:
        prompt = self._prompt
        if isinstance(prompt, AskText) and prompt.validate is not None:
            error = prompt.validate(value)
            if error:
                self._error = error
                return
        self._on_done(value)

    def _submit_select(self, item: SelectItem) -> None:
        index = int(item.value)
        self._on_done(self._choices[index].value)


def _is_filter_char(data: str) -> bool:
    if data in {"\x7f", "\x08"}:
        return True
    return len(data) == 1 and data.isprintable() and data not in {"\n", "\r", "\t"}


async def ask_in_overlay[T](tui: TUI, prompt: Ask[T], styles: Styles) -> T | None:
    """Host a PromptForm inside a running TUI via ``show_overlay``.

    Args:
        tui: The live TUI.
        prompt: Prompt to ask.
        styles: Resolved theme.

    Returns:
        The typed value, or None when the user cancelled.
    """
    if isinstance(prompt, AskSecret):
        return cast(T | None, await asyncio.to_thread(_ask_secret, prompt))
    loop = asyncio.get_running_loop()
    future: asyncio.Future[T | None] = loop.create_future()

    def finish(value: object) -> None:
        if not future.done():
            loop.call_soon_threadsafe(future.set_result, cast(T | None, value))

    form = PromptForm(
        prompt,
        styles,
        on_done=finish,
        on_cancel=lambda: finish(None),
    )
    handle = tui.show_overlay(form, {"anchor": "center", "width": 72})
    handle.focus()
    try:
        return await future
    finally:
        handle.hide()


async def ask_in_transient_tui[T](prompt: Ask[T], styles: Styles) -> T | None:
    """Start a TUI containing only a PromptForm, await, and stop it.

    Args:
        prompt: Prompt to ask.
        styles: Resolved theme.

    Returns:
        The typed value, or None when the user cancelled.
    """
    if isinstance(prompt, AskSecret):
        return cast(T | None, await asyncio.to_thread(_ask_secret, prompt))
    host = FormHost(styles)
    try:
        return await host.ask(prompt)
    finally:
        host.close()


class FormHost:
    """One TUI reused across sequential asks so onboarding does not flicker."""

    def __init__(self, styles: Styles) -> None:
        self._styles = styles
        self._tui = TUI(_default_process_terminal())
        self._started = False

    def start(self) -> None:
        """Start the TUI if needed."""
        if not self._started:
            self._tui.start()
            self._started = True

    def close(self) -> None:
        """Stop the TUI."""
        if self._started:
            self._tui.stop()
            self._started = False

    async def ask[T](self, prompt: Ask[T]) -> T | None:
        """Ask one prompt inside the shared TUI.

        Args:
            prompt: Prompt to ask.

        Returns:
            The typed value, or None when the user cancelled.
        """
        if isinstance(prompt, AskSecret):
            was_started = self._started
            if was_started:
                self._tui.stop()
                self._started = False
            try:
                return cast(T | None, await asyncio.to_thread(_ask_secret, prompt))
            finally:
                if was_started:
                    self.start()
        self.start()
        loop = asyncio.get_running_loop()
        future: asyncio.Future[T | None] = loop.create_future()

        def finish(value: object) -> None:
            if not future.done():
                loop.call_soon_threadsafe(future.set_result, cast(T | None, value))

        form = PromptForm(
            prompt,
            self._styles,
            on_done=finish,
            on_cancel=lambda: finish(None),
        )
        self._tui.children.clear()
        self._tui.add_child(form)
        self._tui.set_focus(form)
        self._tui.request_render()
        return await future


def _ask_secret(prompt: AskSecret) -> str | None:
    try:
        return getpass.getpass(f"{prompt.message} ")
    except (EOFError, KeyboardInterrupt):
        return None
