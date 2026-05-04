"""Streaming query handling for the CLI (pydantic-ai based).

This module uses DisplayManager's LiveMarkdownRenderer to stream Markdown
incrementally as the agent outputs tokens. Tool calls and results are
rendered via DisplayManager helpers.
"""

import asyncio
from collections.abc import Awaitable, Callable
from functools import singledispatchmethod
from typing import Any, AsyncIterable

from pydantic_ai import RunContext
from pydantic_ai.messages import (
    AgentStreamEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
)
from rich.console import Console

from sqlsaber.cli.display import DisplayManager
from sqlsaber.config.logging import get_logger


class StreamingQueryHandler:
    """
    Handles streaming query execution and display using pydantic-ai events.

    Uses DisplayManager.live to render Markdown incrementally as text streams in.
    """

    def __init__(self, console: Console):
        self.console = console
        self.display = DisplayManager(console)
        self.log = get_logger(__name__)
        self._tool_call_names: dict[int, str] = {}

    async def _event_stream_handler(
        self, ctx: RunContext, event_stream: AsyncIterable[AgentStreamEvent]
    ) -> None:
        """
        Handle pydantic-ai streaming events and update Live Markdown via DisplayManager.
        """

        async for event in event_stream:
            messages = getattr(ctx, "messages", None)
            if isinstance(messages, list):
                self.display.set_replay_messages(messages)
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
        elif isinstance(event.part, ToolCallPart):
            self._tool_call_names[event.index] = event.part.tool_name
            self._maybe_start_tool_status(event.part.tool_name)

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
        elif isinstance(d, ToolCallPartDelta):
            if d.tool_name_delta:
                current_name = self._tool_call_names.get(event.index, "")
                updated_name = f"{current_name}{d.tool_name_delta}"
                self._tool_call_names[event.index] = updated_name
                self._maybe_start_tool_status(updated_name)

    @on_event.register
    async def _(self, event: PartEndEvent, ctx: RunContext) -> None:
        self._tool_call_names.pop(event.index, None)

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
            self.display.show_tool_executing(
                event.part.tool_name,
                args,
                tool_call_id=event.part.tool_call_id,
            )
            status = self._tool_execution_status(event.part.tool_name, args)
            if status:
                self.display.live.start_status(status)

    def _maybe_start_tool_status(self, tool_name: str) -> None:
        status = self._tool_execution_status(tool_name, {})
        if status:
            self.display.live.start_status(status)

    def _tool_execution_status(
        self, tool_name: str, args: dict[str, Any]
    ) -> str | None:
        if tool_name == "execute_sql":
            return "Generating SQL..."
        if tool_name == "ask_database":
            database_id = str(args.get("database_id") or "").strip()
            if database_id:
                return f"Querying database {database_id}..."
            return "Querying database..."
        if tool_name == "list_connected_databases":
            return "Inspecting connected databases..."
        if tool_name == "viz":
            return "Generating visualization..."
        return None

    @on_event.register
    async def _(self, event: FunctionToolResultEvent, ctx: RunContext) -> None:
        self.display.live.end_status()
        # Route tool result to appropriate display
        tool_name = event.result.tool_name
        content = event.result.content
        if tool_name is None:
            return
        tool_call_id = getattr(event.result, "tool_call_id", None)
        self.display.show_tool_result(tool_name, content, tool_call_id=tool_call_id)
        # Add a blank line after tool output to separate from next segment
        self.display.show_newline()
        # Show status while agent sends a follow-up request to the model
        self.display.live.start_status("Crunching data...")

    async def execute_streaming_query(
        self,
        user_query: str,
        run_query: Callable[..., Awaitable[Any]],
        cancellation_token: asyncio.Event | None = None,
        message_history: list | None = None,
    ):
        self._tool_call_names.clear()
        self.display.live.prepare_code_blocks()
        try:
            self.log.info("streaming.execute.start")
            self.display.live.start_status("Crunching data...")

            run = await run_query(
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
                self._tool_call_names.clear()
