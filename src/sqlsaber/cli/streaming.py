"""One-shot streaming adapter."""

from __future__ import annotations

from collections.abc import Mapping

from sqlsaber.cli.stream_presenter import AgentStreamPresenter
from sqlsaber.query_results import QueryResultStore
from sqlsaber.render import cli_out
from sqlsaber.render.surface import Surface
from sqlsaber.tools.base import Tool


class StreamingQueryHandler(AgentStreamPresenter):
    """One-shot CLI streaming. Prefer ``AgentStreamPresenter`` at new call sites."""

    def __init__(
        self,
        surface: Surface | None = None,
        display_registry: Mapping[str, Tool] | None = None,
        query_result_store: QueryResultStore | None = None,
    ) -> None:
        super().__init__(
            surface if surface is not None else cli_out(),
            display_registry=display_registry,
            query_result_store=query_result_store,
        )
