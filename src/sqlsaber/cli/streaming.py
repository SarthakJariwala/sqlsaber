"""Streaming query handling for the CLI (pydantic-ai based).

This module uses DisplayManager's LiveMarkdownRenderer to stream Markdown
incrementally as the agent outputs tokens. Tool calls and results are
rendered via DisplayManager helpers.
"""

import asyncio
import json
from functools import singledispatchmethod
from typing import TYPE_CHECKING, AsyncIterable

from pydantic_ai import RunContext
from pydantic_ai.messages import (
    AgentStreamEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
)
from rich.console import Console

from sqlsaber.cli.display import DisplayManager
from sqlsaber.config.logging import get_logger

if TYPE_CHECKING:
    from sqlsaber.agents.pydantic_ai_agent import SQLSaberAgent


class StreamingQueryHandler:
    """
    Handles streaming query execution and display using pydantic-ai events.

    Uses DisplayManager.live to render Markdown incrementally as text streams in.
    """

    def __init__(self, console: Console):
        self.console = console
        self.display = DisplayManager(console)
        self.log = get_logger(__name__)

    async def _event_stream_handler(
        self, ctx: RunContext, event_stream: AsyncIterable[AgentStreamEvent]
    ) -> None:
        """
        Handle pydantic-ai streaming events and update Live Markdown via DisplayManager.
        """

        async for event in event_stream:
            await self.on_event(event, ctx)

    # --- Event routing via singledispatchmethod ---------------------------------------
    @singledispatchmethod
    async def on_event(
        self, event: AgentStreamEvent, ctx: RunContext
    ) -> None:  # default
        return

    @on_event.register
    async def _(self, event: PartStartEvent, ctx: RunContext) -> None:
        if isinstance(event.part, TextPart):
            self.display.live.ensure_segment(TextPart)
            self.display.live.append(event.part.content)
        elif isinstance(event.part, ThinkingPart):
            self.display.live.ensure_segment(ThinkingPart)
            self.display.live.append(event.part.content)

    @on_event.register
    async def _(self, event: PartDeltaEvent, ctx: RunContext) -> None:
        d = event.delta
        if isinstance(d, TextPartDelta):
            delta = d.content_delta or ""
            if delta:
                self.display.live.ensure_segment(TextPart)
                self.display.live.append(delta)
        elif isinstance(d, ThinkingPartDelta):
            delta = d.content_delta or ""
            if delta:
                self.display.live.ensure_segment(ThinkingPart)
                self.display.live.append(delta)

    @on_event.register
    async def _(self, event: FunctionToolCallEvent, ctx: RunContext) -> None:
        # Clear any status/markdown Live so tool output sits between
        self.display.live.end_status()
        self.display.live.end_if_active()
        args = event.part.args_as_dict()

        # Special handling: display SQL via Live as markdown code block
        if event.part.tool_name == "execute_sql":
            query = args.get("query") or ""
            if isinstance(query, str) and query.strip():
                self.display.live.start_sql_block(query)
        else:
            self.display.show_tool_executing(event.part.tool_name, args)

    @on_event.register
    async def _(self, event: FunctionToolResultEvent, ctx: RunContext) -> None:
        # Route tool result to appropriate display
        tool_name = event.result.tool_name
        content = event.result.content
        if tool_name == "list_tables":
            if isinstance(content, (str, dict)):
                self.display.show_table_list(content)
        elif tool_name == "introspect_schema":
            if isinstance(content, (str, dict)):
                self.display.show_schema_info(content)
        elif tool_name == "execute_sql":
            data: dict[str, object] = {}
            if isinstance(content, str):
                try:
                    data = json.loads(content)
                except (json.JSONDecodeError, TypeError) as exc:
                    try:
                        self.console.log(f"Malformed execute_sql result: {exc}")
                    except Exception:
                        pass
            elif isinstance(content, dict):
                data = content

            if isinstance(data, dict):
                if data.get("success") and data.get("results"):
                    self.display.show_query_results(data["results"])  # type: ignore[arg-type]
                elif "error" in data:
                    error_msg = data.get("error")
                    if isinstance(error_msg, str):
                        self.display.show_sql_error(error_msg, data.get("suggestions"))  # type: ignore[arg-type]
        # Add a blank line after tool output to separate from next segment
        self.display.show_newline()
        # Show status while agent sends a follow-up request to the model
        self.display.live.start_status("Crunching data...")

    async def execute_streaming_query(
        self,
        user_query: str,
        sqlsaber_agent: "SQLSaberAgent",
        cancellation_token: asyncio.Event | None = None,
        message_history: list | None = None,
    ):
        self.display.live.prepare_code_blocks()
        try:
            self.log.info("streaming.execute.start")
            self.display.live.start_status("Crunching data...")

            run = await sqlsaber_agent.run(
                user_query,
                message_history=message_history,
                event_stream_handler=self._event_stream_handler,
            )
            self.log.info("streaming.execute.end")
            return run
        except asyncio.CancelledError:
            self.display.show_newline()
            self.console.print("[warning]Query interrupted[/warning]")
            self.log.info("streaming.execute.cancelled")
            return None
        finally:
            try:
                self.display.live.end_status()
            finally:
                self.display.live.end_if_active()
