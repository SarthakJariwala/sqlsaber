"""Buffered streaming query handling for the CLI.

This module buffers content until parts complete, then prints via console.print
instead of using Live rendering. Tool results are displayed as pretty JSON.
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
    PartEndEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
)
from rich.console import Console
from rich.markdown import Markdown

from sqlsaber.config.logging import get_logger

if TYPE_CHECKING:
    from sqlsaber.agents.pydantic_ai_agent import SQLSaberAgent


class BufferedStreamingHandler:
    """
    Non-live streaming handler that buffers content until parts complete.

    Buffers text/thinking content and prints via console.print when the part ends.
    Tool results are displayed as pretty-printed JSON.
    """

    def __init__(self, console: Console):
        self.console = console
        self.log = get_logger(__name__)
        self._buffer: str = ""
        self._current_kind: TextPart | ThinkingPart | None = None

    def _flush_buffer(self) -> None:
        """Print buffered content and reset."""
        if not self._buffer or self._current_kind is None:
            return

        if self._current_kind == TextPart:
            self.console.print(Markdown(self._buffer))
        elif self._current_kind == ThinkingPart:
            self.console.print("[muted]💭 Thinking...[/muted]")
            self.console.print(Markdown(self._buffer, style="muted"))

        self._buffer = ""
        self._current_kind = None

    async def _event_stream_handler(
        self, ctx: RunContext, event_stream: AsyncIterable[AgentStreamEvent]
    ) -> None:
        """Handle pydantic-ai streaming events and buffer content."""
        async for event in event_stream:
            await self.on_event(event, ctx)

    @singledispatchmethod
    async def on_event(self, event: AgentStreamEvent, ctx: RunContext) -> None:
        """Default handler for unregistered event types."""
        return

    @on_event.register
    async def _(self, event: PartStartEvent, ctx: RunContext) -> None:
        """Handle start of a new part (text or thinking)."""
        part = event.part

        if isinstance(part, TextPart):
            new_kind = TextPart
        elif isinstance(part, ThinkingPart):
            new_kind = ThinkingPart
        else:
            self._flush_buffer()
            return

        # Flush if switching from a different kind
        if self._current_kind is not None and self._current_kind != new_kind:
            self._flush_buffer()

        self._current_kind = new_kind
        if part.content:
            self._buffer += part.content

    @on_event.register
    async def _(self, event: PartDeltaEvent, ctx: RunContext) -> None:
        """Handle incremental content delta."""
        d = event.delta
        if isinstance(d, (TextPartDelta, ThinkingPartDelta)):
            delta = d.content_delta or ""
            if delta:
                self._buffer += delta

    @on_event.register
    async def _(self, event: PartEndEvent, ctx: RunContext) -> None:
        """Handle completion of a part by flushing buffered content."""
        self._flush_buffer()

    @on_event.register
    async def _(self, event: FunctionToolCallEvent, ctx: RunContext) -> None:
        """Handle tool call - flush buffers and print tool info."""
        self._flush_buffer()
        self.console.print()
        self.console.print(f"[bold]Executing: {event.part.tool_name}[/bold]")
        args = event.part.args_as_dict()
        if args:
            self.console.print(json.dumps(args, indent=2))

    @on_event.register
    async def _(self, event: FunctionToolResultEvent, ctx: RunContext) -> None:
        """Handle tool result - print as pretty JSON."""
        content = event.result.content
        self.console.print()

        # Parse content if it's a JSON string
        if isinstance(content, str):
            try:
                data = json.loads(content)
                self.console.print(json.dumps(data, indent=2))
            except (json.JSONDecodeError, TypeError):
                self.console.print(content)
        elif isinstance(content, dict):
            self.console.print(json.dumps(content, indent=2))
        else:
            self.console.print(str(content))

        self.console.print()

    async def execute_streaming_query(
        self,
        user_query: str,
        sqlsaber_agent: "SQLSaberAgent",
        cancellation_token: asyncio.Event | None = None,
        message_history: list | None = None,
    ):
        """Execute a query with buffered streaming output."""
        try:
            self.log.info("buffered_streaming.execute.start")

            run = await sqlsaber_agent.run(
                user_query,
                message_history=message_history,
                event_stream_handler=self._event_stream_handler,
            )
            self.log.info("buffered_streaming.execute.end")
            return run
        except asyncio.CancelledError:
            self._flush_buffer()
            self.console.print()
            self.console.print("[warning]Query interrupted[/warning]")
            self.log.info("buffered_streaming.execute.cancelled")
            return None
        except Exception:
            self._flush_buffer()
            self.log.exception("buffered_streaming.execute.error")
            raise
