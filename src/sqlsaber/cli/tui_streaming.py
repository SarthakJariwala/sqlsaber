"""pydantic-ai streaming adapter for the persistent saber-tui chat UI."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from sqlsaber.cli.chat_surface import ChatSurface
from sqlsaber.cli.stream_presenter import AgentStreamPresenter
from sqlsaber.cli.tui_chat import ChatApp
from sqlsaber.query_results import QueryResultStore
from sqlsaber.tools.base import Tool


class TUIStreamingQueryHandler(AgentStreamPresenter):
    """Stream agent output into the persistent chat app."""

    def __init__(
        self,
        app: ChatApp,
        console: Any = None,
        display_registry: Mapping[str, Tool] | None = None,
        *,
        display_registry_provider: Callable[[], Mapping[str, Tool] | None]
        | None = None,
        query_result_store: QueryResultStore | None = None,
    ) -> None:
        super().__init__(
            ChatSurface(app),
            display_registry=display_registry,
            display_registry_provider=display_registry_provider,
            query_result_store=query_result_store,
        )
        self.app = app
        self.console = console
