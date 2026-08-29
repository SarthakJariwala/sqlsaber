"""Non-interactive hosts: TerminalSurface (TTY) and PlainSurface (pipe)."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from typing import TextIO, cast

from sqlsaber.render.blocks import Block, Role
from sqlsaber.render.markdown_text import md_of
from sqlsaber.render.prompts import FormHost, ask_in_transient_tui
from sqlsaber.render.surface import (
    Ask,
    AskConfirm,
    PromptUnavailable,
    TextStream,
)
from sqlsaber.theme.styles import Styles

_MIN_WIDTH = 40
_PLAIN_WIDTH = 80


class _LiveStream:
    def __init__(self, surface: TerminalSurface, role: Role | None) -> None:
        self._surface = surface
        self._role = role
        self._text = ""
        self._closed = False

    def append(self, chunk: str) -> None:
        if self._closed:
            return
        self._text += chunk
        self._surface._repaint_live(self._text, role=self._role)

    def set(self, text: str) -> None:
        if self._closed:
            return
        self._text = text
        self._surface._repaint_live(self._text, role=self._role)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._surface._settle_live()

    def discard(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._surface._erase_live()


class _BufferedStream:
    def __init__(self, surface: PlainSurface) -> None:
        self._surface = surface
        self._chunks: list[str] = []
        self._closed = False

    def append(self, chunk: str) -> None:
        if not self._closed:
            self._chunks.append(chunk)

    def set(self, text: str) -> None:
        if not self._closed:
            self._chunks = [text]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        text = "".join(self._chunks)
        if text:
            if not text.endswith("\n"):
                text += "\n"
            self._surface._write(text)

    def discard(self) -> None:
        self._closed = True
        self._chunks.clear()


class TerminalSurface:
    """Styled output to a TTY with no event loop running.

    Streams repaint in place. An open live region is finalized before
    ``emit`` so output order matches call order. Prompts start a short-lived
    TUI, or reuse ``FormHost`` when a form session is active.
    """

    def __init__(self, stream: TextIO, styles: Styles) -> None:
        self._stream = stream
        self._styles = styles
        self._live_lines = 0
        self._open_stream: _LiveStream | None = None
        self._form: FormHost | None = None

    def start_form(self) -> None:
        """Open a shared TUI."""
        if self._form is None:
            self._form = FormHost(self._styles)
            self._form.start()

    def end_form(self) -> None:
        """Stop the shared TUI."""
        if self._form is not None:
            self._form.close()
            self._form = None

    def emit(self, *blocks: Block) -> None:
        """Write finished blocks, finalizing any live region first.

        Args:
            blocks: Blocks to render.
        """
        self._finalize_live()
        from sqlsaber.render.tui_blocks import components_for

        components = components_for(blocks, self._styles)
        width = self._width()
        lines: list[str] = []
        for component in components:
            rendered = component.render(width)
            if lines and rendered:
                lines.append("")
            lines.extend(rendered)
        if lines:
            self._write("\n".join(lines) + "\n")

    def stream(
        self,
        *,
        role: Role | None = None,
        replace: TextStream | None = None,
        before: TextStream | None = None,
    ) -> TextStream:
        """Open an in-place markdown stream.

        Args:
            role: Optional tint for the streamed markdown.
            replace: Existing stream to discard before opening this one.
            before: Ignored; this host has one live region.

        Returns:
            A live stream handle.
        """
        del before
        if replace is not None:
            replace.discard()
        else:
            self._finalize_live()
        live = _LiveStream(self, role)
        self._open_stream = live
        return live

    def status(
        self, text: str | None, *, on_cancel: Callable[[], None] | None = None
    ) -> None:
        """Show or clear a single status line.

        Args:
            text: Status text, or None to clear.
            on_cancel: Unused on this host; cancellation is Ctrl-C.
        """
        del on_cancel
        self._finalize_live()
        if text is None:
            return
        from sqlsaber.render.tui_blocks import components_for
        from sqlsaber.render.blocks import note

        components = components_for([note(text, role="warning")], self._styles)
        lines: list[str] = []
        width = self._width()
        for component in components:
            lines.extend(component.render(width))
        self._write_live(lines)

    async def ask[T](self, prompt: Ask[T]) -> T | None:
        """Ask a prompt on this TTY.

        Args:
            prompt: Prompt to ask.

        Returns:
            The typed value, or None when cancelled.
        """
        if isinstance(prompt, AskConfirm) and prompt.assume_yes:
            return cast(T, True)
        if self._form is not None:
            return await self._form.ask(prompt)
        return await ask_in_transient_tui(prompt, self._styles)

    def _width(self) -> int:
        try:
            columns = shutil.get_terminal_size().columns
        except OSError:
            columns = _PLAIN_WIDTH
        return max(_MIN_WIDTH, columns)

    def _screen_rows(self) -> int:
        try:
            return max(4, shutil.get_terminal_size().lines)
        except OSError:
            return 24

    def _finalize_live(self) -> None:
        if self._open_stream is not None:
            self._open_stream.close()
            self._open_stream = None
        elif self._live_lines:
            self._erase_live()

    def _repaint_live(self, text: str, *, role: Role | None) -> None:
        from sqlsaber.render.blocks import md
        from sqlsaber.render.tui_blocks import components_for

        cleaned = "".join(
            char
            for char in text
            if char in "\n\t" or (ord(char) >= 0x20 and not (0x7F <= ord(char) < 0xA0))
        )
        components = components_for([md(cleaned, role=role)], self._styles)
        width = self._width()
        lines: list[str] = []
        for component in components:
            lines.extend(component.render(width))
        max_live = max(1, self._screen_rows() - 2)
        if len(lines) > max_live:
            lines = lines[-max_live:]
        self._write_live(lines)

    def _write_live(self, lines: list[str]) -> None:
        self._erase_live()
        if not lines:
            return
        self._write("\n".join(lines) + "\n")
        self._live_lines = len(lines)

    def _erase_live(self) -> None:
        if self._live_lines <= 0:
            return
        self._write(f"\x1b[{self._live_lines}A\r\x1b[J")
        self._live_lines = 0

    def _settle_live(self) -> None:
        self._live_lines = 0
        self._open_stream = None

    def _write(self, text: str) -> None:
        self._stream.write(text)
        self._stream.flush()


class PlainSurface:
    """Unstyled markdown to a pipe. No ANSI ever leaves here."""

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream
        self._open: _BufferedStream | None = None

    def emit(self, *blocks: Block) -> None:
        """Write ``md_of(blocks)`` plus a trailing newline.

        Args:
            blocks: Blocks to serialize.
        """
        if self._open is not None:
            self._open.close()
            self._open = None
        text = md_of(blocks)
        if text:
            if not text.endswith("\n"):
                text += "\n"
            self._write(text)

    def stream(
        self,
        *,
        role: Role | None = None,
        replace: TextStream | None = None,
        before: TextStream | None = None,
    ) -> TextStream:
        """Buffer markdown and write once on close.

        Args:
            role: Ignored on a pipe.
            replace: Existing stream to discard before opening this one.
            before: Ignored on a pipe.

        Returns:
            A buffered stream handle.
        """
        del role, before
        if replace is not None:
            replace.discard()
        elif self._open is not None:
            self._open.close()
        buffered = _BufferedStream(self)
        self._open = buffered
        return buffered

    def status(
        self, text: str | None, *, on_cancel: Callable[[], None] | None = None
    ) -> None:
        """No-op on a pipe."""
        del text, on_cancel

    async def ask[T](self, prompt: Ask[T]) -> T | None:
        """``assume_yes`` short-circuits; otherwise raise PromptUnavailable.

        Args:
            prompt: Prompt to ask.

        Returns:
            True when ``AskConfirm.assume_yes`` is set.

        Raises:
            PromptUnavailable: When the prompt needs a TTY.
        """
        if isinstance(prompt, AskConfirm) and prompt.assume_yes:
            return cast(T, True)
        raise PromptUnavailable(prompt)

    def _write(self, text: str) -> None:
        self._stream.write(text)
        self._stream.flush()
