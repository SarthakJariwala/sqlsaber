"""One-shot streaming adapter. Constructs a Surface from the legacy Console."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from typing import Any

from sqlsaber.cli.stream_presenter import AgentStreamPresenter
from sqlsaber.query_results import QueryResultStore
from sqlsaber.render.surface import Surface
from sqlsaber.tools.base import Tool


def surface_from_console(console: Any) -> Surface:
    """TTY vs pipe surface from a Rich Console (transitional).

    Args:
        console: Legacy Rich console. ``file`` and ``is_terminal`` are read.

    Returns:
        ``TerminalSurface`` or ``PlainSurface``.
    """
    stream = getattr(console, "file", None) or sys.stdout
    is_tty = bool(getattr(console, "is_terminal", False))
    if is_tty:
        from sqlsaber.render.terminal import TerminalSurface
        from sqlsaber.theme.styles import get_styles

        return TerminalSurface(stream, get_styles())
    from sqlsaber.render.terminal import PlainSurface

    return PlainSurface(stream)


class StreamingQueryHandler(AgentStreamPresenter):
    """One-shot CLI streaming. Prefer ``AgentStreamPresenter`` at new call sites."""

    def __init__(
        self,
        console: Any,
        display_registry: Mapping[str, Tool] | None = None,
        query_result_store: QueryResultStore | None = None,
    ) -> None:
        super().__init__(
            surface_from_console(console),
            display_registry=display_registry,
            query_result_store=query_result_store,
        )
        self.console = console
