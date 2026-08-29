"""ChatApp-backed Surface."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from saber_tui.components import Markdown

from sqlsaber.render.blocks import Block, Role
from sqlsaber.render.surface import Ask, AskConfirm, TextStream
from sqlsaber.theme.styles import get_styles
from sqlsaber.utils.text_input import sanitize_terminal_text

if TYPE_CHECKING:
    from sqlsaber.cli.tui_chat import ChatApp

STREAM_FLUSH_INTERVAL_SECONDS = 1 / 30


class _ChatStream:
    def __init__(self, surface: ChatSurface, component: Markdown) -> None:
        self._surface = surface
        self.component = component
        self._closed = False

    def append(self, chunk: str) -> None:
        if self._closed or not chunk:
            return
        self.component.append_text(sanitize_terminal_text(chunk))
        self._surface._paint()

    def set(self, text: str) -> None:
        if self._closed:
            return
        self.component.set_text(sanitize_terminal_text(text))
        self._surface._paint()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._surface.app.tui.flush_render()
        self._surface.app.freeze_markdown(self.component)

    def discard(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._surface.app.remove_markdown(self.component)


class ChatSurface:
    """Print, stream, status, and prompt on the persistent chat TUI."""

    def __init__(self, app: ChatApp) -> None:
        self.app = app
        self._last_flush = 0.0

    def emit(self, *blocks: Block) -> None:
        """Append finished blocks as saber-tui components.

        Args:
            blocks: Blocks to render. An empty sequence is a no-op.
        """
        if not blocks:
            return
        from sqlsaber.render.tui_blocks import components_for

        styles = get_styles()
        for component in components_for(blocks, styles, tui=self.app.tui):
            self.app._append_component(component)

    def stream(
        self,
        *,
        role: Role | None = None,
        replace: TextStream | None = None,
        before: TextStream | None = None,
    ) -> TextStream:
        """Open a Markdown component that grows in place.

        Args:
            role: ``muted`` uses the thinking style.
            replace: Stream whose component is swapped in place.
            before: Stream to insert this component ahead of.

        Returns:
            A chat-backed text stream.
        """
        muted = role == "muted"
        if isinstance(replace, _ChatStream):
            component = self.app.replace_markdown(replace.component, "", muted=muted)
            replace._closed = True
        elif isinstance(before, _ChatStream):
            component = self.app.insert_markdown_before(before.component, muted=muted)
        else:
            component = self.app.append_markdown("", muted=muted)
        return _ChatStream(self, component)

    def status(
        self, text: str | None, *, on_cancel: Callable[[], None] | None = None
    ) -> None:
        """Show or clear the chat loader.

        Args:
            text: Status text, or None to clear.
            on_cancel: Unused; ChatApp owns the cancel callback.
        """
        del on_cancel
        if text is None:
            self.app.clear_status()
            return
        self.app.set_loading(text)

    async def ask[T](self, prompt: Ask[T]) -> T | None:
        """Ask on the live TUI. ``assume_yes`` short-circuits confirms.

        Args:
            prompt: Prompt to ask.

        Returns:
            The typed value, or None when cancelled.
        """
        if isinstance(prompt, AskConfirm) and prompt.assume_yes:
            return cast(T, True)
        from sqlsaber.render.prompts import ask_in_transient_tui

        return await ask_in_transient_tui(prompt, get_styles())

    def _paint(self) -> None:
        self.app.tui.request_render()
        now = time.monotonic()
        if now - self._last_flush >= STREAM_FLUSH_INTERVAL_SECONDS:
            self.app.tui.flush_render()
            self._last_flush = now


def chat_surface(app: Any) -> ChatSurface:
    """Build a ``ChatSurface`` for ``app``.

    Args:
        app: Persistent ``ChatApp``.

    Returns:
        A surface bound to that app.
    """
    return ChatSurface(app)
