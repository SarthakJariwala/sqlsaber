"""pydantic-ai streaming adapter for the persistent saber-tui chat UI."""

import asyncio
from collections.abc import Awaitable, Callable
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
from rich.markdown import Markdown

from sqlsaber.cli.display import DisplayManager
from sqlsaber.cli.tui_chat import ChatApp
from sqlsaber.config.logging import get_logger
from sqlsaber.theme.manager import get_theme_manager


class _QueryInterrupted(Exception):
    """Internal signal for user-requested query interruption."""


class TUIStreamingQueryHandler:
    """Stream agent output into the persistent chat app."""

    def __init__(self, app: ChatApp, console: Console):
        self.app = app
        self.console = console
        self.log = get_logger(__name__)
        self._tool_call_names: dict[int, str] = {}
        self._stream_component = None
        self._stream_kind: type | None = None
        self._stream_buffer = ""
        self._replay_messages: list | None = None
        self._cancellation_token: asyncio.Event | None = None

    async def _event_stream_handler(
        self, ctx: RunContext, event_stream: AsyncIterable[AgentStreamEvent]
    ) -> None:
        self._raise_if_cancelled()
        async for event in event_stream:
            self._raise_if_cancelled()
            messages = getattr(ctx, "messages", None)
            if isinstance(messages, list):
                self._replay_messages = messages
            await self.on_event(event)
            self._raise_if_cancelled()

    async def on_event(self, event: AgentStreamEvent) -> None:
        if isinstance(event, PartStartEvent):
            self._on_part_start(event)
        elif isinstance(event, PartDeltaEvent):
            self._on_part_delta(event)
        elif isinstance(event, PartEndEvent):
            self._finish_stream_segment()
            self._tool_call_names.pop(event.index, None)
        elif isinstance(event, FunctionToolCallEvent):
            self._on_tool_call(event)
        elif isinstance(event, FunctionToolResultEvent):
            self._on_tool_result(event)

    def _on_part_start(self, event: PartStartEvent) -> None:
        if isinstance(event.part, TextPart):
            self._append_stream(TextPart, event.part.content)
        elif isinstance(event.part, ThinkingPart):
            self._append_stream(ThinkingPart, event.part.content)
        elif isinstance(event.part, ToolCallPart):
            self._tool_call_names[event.index] = event.part.tool_name
            self._maybe_start_sql_generation_status(event.part.tool_name)

    def _on_part_delta(self, event: PartDeltaEvent) -> None:
        delta = event.delta
        if isinstance(delta, TextPartDelta):
            self._append_stream(TextPart, delta.content_delta or "")
        elif isinstance(delta, ThinkingPartDelta):
            self._append_stream(ThinkingPart, delta.content_delta or "")
        elif isinstance(delta, ToolCallPartDelta) and delta.tool_name_delta:
            current_name = self._tool_call_names.get(event.index, "")
            updated_name = f"{current_name}{delta.tool_name_delta}"
            self._tool_call_names[event.index] = updated_name
            self._maybe_start_sql_generation_status(updated_name)

    def _on_tool_call(self, event: FunctionToolCallEvent) -> None:
        self._finish_stream_segment()
        args = event.part.args_as_dict()
        if event.part.tool_name == "execute_sql":
            query = args.get("query") or ""
            if isinstance(query, str) and query.strip():
                tm = get_theme_manager()
                self.app.append_rich(
                    lambda console: console.print(
                        Markdown(
                            f"```sql\n{query}\n```",
                            code_theme=tm.pygments_style_name,
                        )
                    )
                )
            return

        self._append_display(
            lambda display: display.show_tool_executing(event.part.tool_name, args)
        )
        if event.part.tool_name == "viz":
            self.app.set_loading("Generating visualization...")

    def _on_tool_result(self, event: FunctionToolResultEvent) -> None:
        self.app.clear_status()
        tool_name = event.part.tool_name
        if tool_name is None:
            return
        self._append_display(
            lambda display: display.show_tool_result(tool_name, event.part.content)
        )
        self.app.set_loading("Crunching data...")

    def _append_stream(self, kind: type, text: str) -> None:
        if not text:
            return
        if self._stream_component is None or self._stream_kind is not kind:
            self._finish_stream_segment()
            self._stream_kind = kind
            self._stream_buffer = ""
            self._stream_component = self.app.append_markdown(
                "", muted=kind is ThinkingPart
            )
        self._stream_buffer += text
        self._stream_component.set_text(self._stream_buffer)
        self.app.tui.request_render()

    def _finish_stream_segment(self) -> None:
        if self._stream_component is not None:
            self.app.freeze_markdown(self._stream_component)
        self._stream_component = None
        self._stream_kind = None
        self._stream_buffer = ""

    def _maybe_start_sql_generation_status(self, tool_name: str) -> None:
        if tool_name == "execute_sql":
            self.app.set_loading("Generating SQL...")

    def _append_display(self, render: Callable[[DisplayManager], None]) -> None:
        def capture(console: Console) -> None:
            display = DisplayManager(console)
            if self._replay_messages is not None:
                display.set_replay_messages(self._replay_messages)
            render(display)

        self.app.append_rich(capture)

    def _raise_if_cancelled(self) -> None:
        if self._cancellation_token is not None and self._cancellation_token.is_set():
            raise _QueryInterrupted

    async def execute_streaming_query(
        self,
        user_query: str,
        run_query: Callable[..., Awaitable[Any]],
        cancellation_token: asyncio.Event | None = None,
        message_history: list | None = None,
    ):
        self._tool_call_names.clear()
        self._finish_stream_segment()
        self._cancellation_token = cancellation_token
        try:
            self.log.info("streaming.execute.start")
            self.app.set_loading("Crunching data...")
            run = await run_query(
                user_query,
                message_history=message_history,
                event_stream_handler=self._event_stream_handler,
            )
            self.log.info("streaming.execute.end")
            return run
        except _QueryInterrupted:
            self.app.append_rich(
                lambda console: console.print("[warning]Query interrupted[/warning]")
            )
            self.log.info("streaming.execute.interrupted")
            return None
        except asyncio.CancelledError:
            self.app.append_rich(
                lambda console: console.print("[warning]Query interrupted[/warning]")
            )
            self.log.info("streaming.execute.cancelled")
            return None
        finally:
            self.app.clear_status()
            self._tool_call_names.clear()
            self._cancellation_token = None
