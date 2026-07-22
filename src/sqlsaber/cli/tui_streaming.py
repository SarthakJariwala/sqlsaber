"""pydantic-ai streaming adapter for the persistent saber-tui chat UI."""

import asyncio
import json
import time
from collections.abc import Awaitable, Callable, Mapping
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
from pydantic_core import from_json
from rich.console import Console
from saber_tui.components import Markdown

from sqlsaber.cli.display import DisplayManager
from sqlsaber.cli.tui_chat import ChatApp
from sqlsaber.config.logging import get_logger
from sqlsaber.query_result_resolution import (
    QueryResultReference,
    query_result_context_from_run,
    query_result_from_metadata,
    resolve_query_result,
)
from sqlsaber.query_results import QueryResultStore, QueryResultUnavailable
from sqlsaber.tools.base import Tool
from sqlsaber.utils.text_input import sanitize_terminal_text


def _fallback_sql_markdown(content: object) -> str | None:
    """Minimal fallback for embedding tests without a managed display registry."""
    try:
        payload = json.loads(content) if isinstance(content, str) else content
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    value: dict[str, Any] = {str(key): item for key, item in payload.items()}
    rows = value.get("results")
    if not isinstance(rows, list):
        return "✓ Query completed successfully" if value.get("success") else None
    if not rows:
        return "*0 rows returned*"
    normalized = [row if isinstance(row, dict) else {"value": row} for row in rows]
    columns = list(dict.fromkeys(key for row in normalized for key in row))
    lines = [
        f"**Results ({len(normalized)} rows):**",
        "",
        "| " + " | ".join(str(column) for column in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(row.get(column, "")) for column in columns) + " |"
        for row in normalized[:20]
    )
    return "\n".join(lines)


class _QueryInterrupted(Exception):
    """Internal signal for user-requested query interruption."""


STREAM_FLUSH_INTERVAL_SECONDS = 1 / 30


def _partial_json_query(args: str) -> str | None:
    """Decode the complete portion of a query value from partial JSON arguments."""
    try:
        parsed = from_json(args, allow_partial="trailing-strings")
    except ValueError:
        return None
    if not isinstance(parsed, dict):
        return None
    query = parsed.get("query")
    return query if isinstance(query, str) else None


class TUIStreamingQueryHandler:
    """Stream agent output into the persistent chat app."""

    def __init__(
        self,
        app: ChatApp,
        console: Console,
        display_registry: Mapping[str, Tool] | None = None,
        *,
        display_registry_provider: Callable[[], Mapping[str, Tool] | None]
        | None = None,
        query_result_store: QueryResultStore | None = None,
    ):
        self.app = app
        self.console = console
        self.log = get_logger(__name__)
        self._tool_call_names: dict[int, str] = {}
        self._tool_call_args: dict[int, str | dict[str, Any]] = {}
        self._tool_call_ids: dict[int, str] = {}
        self._sql_stream_components: dict[int, Markdown] = {}
        self._sql_stream_queries: dict[int, str] = {}
        self._stream_components: dict[int, Markdown] = {}
        self._stream_kinds: dict[int, type] = {}
        self._finished_stream_indexes: set[int] = set()
        self._last_stream_render_at = 0.0
        self._replay_messages: list | None = None
        self._cancellation_token: asyncio.Event | None = None
        self._display_registry = display_registry
        self._display_registry_provider = display_registry_provider
        self.query_result_store = query_result_store

    async def _event_stream_handler(
        self, ctx: RunContext, event_stream: AsyncIterable[AgentStreamEvent]
    ) -> None:
        try:
            self._raise_if_cancelled()
            async for event in event_stream:
                self._raise_if_cancelled()
                messages = getattr(ctx, "messages", None)
                if isinstance(messages, list):
                    self._replay_messages = messages
                await self.on_event(event, ctx)
                self._raise_if_cancelled()
        finally:
            self._reset_response_stream_state()

    async def on_event(
        self, event: AgentStreamEvent, ctx: RunContext | None = None
    ) -> None:
        if isinstance(event, PartStartEvent):
            self._on_part_start(event)
        elif isinstance(event, PartDeltaEvent):
            self._on_part_delta(event)
        elif isinstance(event, PartEndEvent):
            if isinstance(event.part, TextPart | ThinkingPart):
                self._finish_stream_segment(event.index)
            elif isinstance(event.part, ToolCallPart):
                self._tool_call_names[event.index] = event.part.tool_name
                self._tool_call_ids[event.index] = event.part.tool_call_id
                self._set_tool_call_args(event.index, event.part.args)
        elif isinstance(event, FunctionToolCallEvent):
            self._on_tool_call(event)
        elif isinstance(event, FunctionToolResultEvent):
            await self._on_tool_result(event, ctx)

    def _on_part_start(self, event: PartStartEvent) -> None:
        previous = self._take_replaced_component(event.index)
        if isinstance(event.part, TextPart):
            self._start_stream(
                event.index, TextPart, event.part.content, previous=previous
            )
        elif isinstance(event.part, ThinkingPart):
            self._start_stream(
                event.index, ThinkingPart, event.part.content, previous=previous
            )
        elif isinstance(event.part, ToolCallPart):
            self._tool_call_names[event.index] = event.part.tool_name
            self._tool_call_ids[event.index] = event.part.tool_call_id
            if event.part.tool_name == "execute_sql":
                if previous is not None:
                    replacement = self.app.replace_markdown(previous)
                    self._sql_stream_components[event.index] = replacement
                else:
                    self._ensure_sql_stream(event.index)
            elif previous is not None:
                self.app.remove_markdown(previous)
            self._set_tool_call_args(event.index, event.part.args)
            self._maybe_start_sql_generation_status(event.part.tool_name)
        elif previous is not None:
            self.app.remove_markdown(previous)

    def _on_part_delta(self, event: PartDeltaEvent) -> None:
        delta = event.delta
        if isinstance(delta, TextPartDelta):
            self._append_stream(event.index, TextPart, delta.content_delta or "")
        elif isinstance(delta, ThinkingPartDelta):
            self._append_stream(event.index, ThinkingPart, delta.content_delta or "")
        elif isinstance(delta, ToolCallPartDelta):
            if delta.tool_name_delta:
                current_name = self._tool_call_names.get(event.index, "")
                updated_name = f"{current_name}{delta.tool_name_delta}"
                self._tool_call_names[event.index] = updated_name
                self._maybe_start_sql_generation_status(updated_name)
                if updated_name == "execute_sql":
                    self._ensure_sql_stream(event.index)
            if delta.tool_call_id:
                self._tool_call_ids[event.index] = delta.tool_call_id
            self._append_tool_call_args(event.index, delta.args_delta)
            self._update_streamed_sql_from_args(event.index)

    def _on_tool_call(self, event: FunctionToolCallEvent) -> None:
        self._finish_all_stream_segments()
        args = event.part.args_as_dict()
        index = next(
            (
                index
                for index, tool_call_id in self._tool_call_ids.items()
                if tool_call_id == event.part.tool_call_id
            ),
            None,
        )
        if event.part.tool_name == "execute_sql":
            if event.args_valid is False:
                if index is not None:
                    self._discard_sql_stream(index)
                    self._clear_tool_call_state(index)
                return
            query = args.get("query") or ""
            if isinstance(query, str) and query.strip():
                if index is None:
                    self._append_complete_sql(query)
                else:
                    self._update_sql_stream(index, query)
            elif index is not None:
                self._discard_sql_stream(index)
            if index is not None:
                self._clear_tool_call_state(index)
            return

        self._append_display(
            lambda display: display.show_tool_executing(event.part.tool_name, args)
        )
        if index is not None:
            self._discard_sql_stream(index)
            self._clear_tool_call_state(index)
        if event.part.tool_name == "viz":
            self.app.set_loading("Generating visualization...")

    async def _on_tool_result(
        self, event: FunctionToolResultEvent, ctx: RunContext | None
    ) -> None:
        tool_name = event.part.tool_name
        if tool_name is None:
            self.app.clear_status()
            return
        content = event.part.content
        complete_unavailable = False
        if (
            tool_name == "execute_sql"
            and ctx is not None
            and self.query_result_store is not None
        ):
            descriptor = query_result_from_metadata(
                getattr(event.part, "metadata", None)
            )
            if descriptor is not None:
                reference = QueryResultReference(
                    tool_call_id=event.part.tool_call_id,
                    file=descriptor.file,
                    descriptor=descriptor,
                )
                try:
                    resolved = await resolve_query_result(
                        reference,
                        store=self.query_result_store,
                        context=query_result_context_from_run(ctx),
                    )
                    content = resolved.data.decode("utf-8")
                except (QueryResultUnavailable, UnicodeDecodeError):
                    complete_unavailable = True
        if tool_name == "execute_sql":
            registry = self._resolve_display_registry() or {}
            renderer = registry.get("execute_sql")
            render_markdown = getattr(renderer, "render_result_markdown", None)
            markdown = (
                render_markdown(content)
                if callable(render_markdown)
                else _fallback_sql_markdown(content)
            )
            if markdown is not None:
                if complete_unavailable:
                    markdown += "\n\n*Complete result unavailable; showing preview.*"
                self.app.append_markdown(markdown)
                self.app.set_loading("Crunching data...")
                return
        self._append_display(
            lambda display: display.show_tool_result(
                tool_name,
                content,
                tool_call_id=event.part.tool_call_id,
                metadata=getattr(event.part, "metadata", None),
            )
        )
        self.app.set_loading("Crunching data...")

    def _start_stream(
        self,
        index: int,
        kind: type,
        text: str,
        *,
        previous: Markdown | None = None,
    ) -> None:
        self._finished_stream_indexes.discard(index)
        if previous is not None:
            component = self.app.replace_markdown(
                previous, "", muted=kind is ThinkingPart
            )
        else:
            component = self.app.append_markdown("", muted=kind is ThinkingPart)
        self._stream_components[index] = component
        self._stream_kinds[index] = kind
        self._append_stream(index, kind, text)

    def _append_stream(self, index: int, kind: type, text: str) -> None:
        if not text:
            return
        component = self._stream_components.get(index)
        if component is None or self._stream_kinds.get(index) is not kind:
            self._start_stream(index, kind, "")
            component = self._stream_components[index]
        component.append_text(sanitize_terminal_text(text))
        self._render_stream_frame()

    def _take_replaced_component(self, index: int) -> Markdown | None:
        response_component = self._stream_components.pop(index, None)
        sql_component = self._sql_stream_components.pop(index, None)
        self._stream_kinds.pop(index, None)
        self._finished_stream_indexes.discard(index)
        self._tool_call_names.pop(index, None)
        self._tool_call_args.pop(index, None)
        self._tool_call_ids.pop(index, None)
        self._sql_stream_queries.pop(index, None)

        if (
            response_component is not None
            and sql_component is not None
            and response_component is not sql_component
        ):
            self.app.remove_markdown(sql_component)
        return response_component or sql_component

    def _set_tool_call_args(
        self, index: int, args: str | dict[str, Any] | None
    ) -> None:
        if isinstance(args, str):
            self._tool_call_args[index] = args
        elif isinstance(args, dict):
            self._tool_call_args[index] = dict(args)
        else:
            return
        self._update_streamed_sql_from_args(index)

    def _append_tool_call_args(
        self, index: int, delta: str | dict[str, Any] | None
    ) -> None:
        if isinstance(delta, str):
            current = self._tool_call_args.get(index, "")
            if isinstance(current, str):
                self._tool_call_args[index] = current + delta
        elif isinstance(delta, dict):
            current = self._tool_call_args.get(index, {})
            if isinstance(current, dict):
                self._tool_call_args[index] = {**current, **delta}
        else:
            return

    def _update_streamed_sql_from_args(self, index: int) -> None:
        if self._tool_call_names.get(index) != "execute_sql":
            return
        args = self._tool_call_args.get(index)
        if isinstance(args, str):
            query = _partial_json_query(args)
        elif isinstance(args, dict):
            value = args.get("query")
            query = value if isinstance(value, str) else None
        else:
            query = None
        if query:
            self._update_sql_stream(index, query)

    def _update_sql_stream(self, index: int, query: str) -> None:
        safe_query = sanitize_terminal_text(query)
        if self._sql_stream_queries.get(index) == safe_query:
            return
        component = self._ensure_sql_stream(index)
        component.set_text(self._sql_markdown(safe_query))
        self._sql_stream_queries[index] = safe_query
        self._render_stream_frame()

    def _append_complete_sql(self, query: str) -> None:
        safe_query = sanitize_terminal_text(query)
        self.app.append_markdown(self._sql_markdown(safe_query))

    def _ensure_sql_stream(self, index: int) -> Markdown:
        component = self._sql_stream_components.get(index)
        if component is not None:
            return component
        later_indexes = sorted(
            candidate for candidate in self._sql_stream_components if candidate > index
        )
        if later_indexes:
            component = self.app.insert_markdown_before(
                self._sql_stream_components[later_indexes[0]]
            )
        else:
            component = self.app.append_markdown()
        self._sql_stream_components[index] = component
        return component

    @staticmethod
    def _sql_markdown(query: str) -> str:
        longest_run = 0
        current_run = 0
        for char in query:
            current_run = current_run + 1 if char == "`" else 0
            longest_run = max(longest_run, current_run)
        fence = "`" * max(3, longest_run + 1)
        return f"{fence}sql\n{query}\n{fence}"

    def _render_stream_frame(self) -> None:
        self.app.tui.request_render()
        now = time.monotonic()
        if now - self._last_stream_render_at >= STREAM_FLUSH_INTERVAL_SECONDS:
            self.app.tui.flush_render()
            self._last_stream_render_at = now

    def _clear_tool_call_state(self, index: int) -> None:
        self._tool_call_names.pop(index, None)
        self._tool_call_args.pop(index, None)
        self._tool_call_ids.pop(index, None)
        self._sql_stream_components.pop(index, None)
        self._sql_stream_queries.pop(index, None)

    def _discard_sql_stream(self, index: int) -> None:
        component = self._sql_stream_components.get(index)
        if component is not None:
            self.app.remove_markdown(component)

    def _reset_tool_call_state(self, *, remove_previews: bool = False) -> None:
        if remove_previews:
            for component in list(self._sql_stream_components.values()):
                self.app.remove_markdown(component)
        self._tool_call_names.clear()
        self._tool_call_args.clear()
        self._tool_call_ids.clear()
        self._sql_stream_components.clear()
        self._sql_stream_queries.clear()

    def _finish_stream_segment(self, index: int) -> None:
        component = self._stream_components.get(index)
        if component is not None and index not in self._finished_stream_indexes:
            self.app.tui.flush_render()
            self.app.freeze_markdown(component)
            self._finished_stream_indexes.add(index)

    def _finish_all_stream_segments(self) -> None:
        for index in list(self._stream_components):
            self._finish_stream_segment(index)

    def _reset_response_stream_state(self) -> None:
        self._stream_components.clear()
        self._stream_kinds.clear()
        self._finished_stream_indexes.clear()

    def _maybe_start_sql_generation_status(self, tool_name: str) -> None:
        if tool_name == "execute_sql":
            self.app.set_loading("Generating SQL...")

    def _resolve_display_registry(self) -> Mapping[str, Tool] | None:
        if self._display_registry_provider is not None:
            return self._display_registry_provider()
        return self._display_registry

    def _append_display(self, render: Callable[[DisplayManager], None]) -> None:
        def capture(console: Console) -> None:
            display = DisplayManager(console, self._resolve_display_registry())
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
        self._reset_tool_call_state(remove_previews=True)
        self._finish_all_stream_segments()
        self._last_stream_render_at = 0.0
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
            self._reset_tool_call_state(remove_previews=True)
            self._reset_response_stream_state()
            self._cancellation_token = None
