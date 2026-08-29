"""One streaming presenter for TTY, pipes, and the persistent chat TUI."""

from __future__ import annotations

import asyncio
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

from sqlsaber.config.logging import get_logger
from sqlsaber.query_result_resolution import (
    QueryResultReference,
    query_result_context_from_run,
    query_result_from_metadata,
    resolve_query_result,
)
from sqlsaber.query_results import QueryResultStore, QueryResultUnavailable
from sqlsaber.render.blocks import Role, code, md, warn
from sqlsaber.render.markdown_text import markdown_source
from sqlsaber.render.surface import Surface, TextStream
from sqlsaber.tools.base import Tool
from sqlsaber.tools.renderer import (
    ToolRenderContext,
    ToolRenderer,
    core_display_registry,
)
from sqlsaber.utils.text_input import sanitize_terminal_text


class _QueryInterrupted(Exception):
    """Internal signal for user-requested query interruption."""


class AgentStreamPresenter:
    """Route pydantic-ai stream events onto one ``Surface``."""

    def __init__(
        self,
        surface: Surface,
        *,
        display_registry: Mapping[str, Tool] | None = None,
        display_registry_provider: Callable[[], Mapping[str, Tool] | None]
        | None = None,
        query_result_store: QueryResultStore | None = None,
    ) -> None:
        self.surface = surface
        self.log = get_logger(__name__)
        self.query_result_store = query_result_store
        self._display_registry = display_registry
        self._display_registry_provider = display_registry_provider
        self._replay_messages: list | None = None
        self._cancellation_token: asyncio.Event | None = None
        self._tool_call_names: dict[int, str] = {}
        self._tool_call_args: dict[int, str | dict[str, Any]] = {}
        self._tool_call_ids: dict[int, str] = {}
        self._text_streams: dict[int, TextStream] = {}
        self._sql_streams: dict[int, TextStream] = {}
        self._sql_stream_queries: dict[int, str] = {}
        self._stream_kinds: dict[int, type] = {}
        self._finished_stream_indexes: set[int] = set()

    @property
    def _stream_components(self) -> dict[int, Any]:
        found: dict[int, Any] = {}
        for index, stream in self._text_streams.items():
            component = getattr(stream, "component", None)
            if component is not None:
                found[index] = component
        return found

    @property
    def _sql_stream_components(self) -> dict[int, Any]:
        found: dict[int, Any] = {}
        for index, stream in self._sql_streams.items():
            component = getattr(stream, "component", None)
            if component is not None:
                found[index] = component
        return found

    def set_replay_messages(self, messages: list) -> None:
        """Record message history for tool replay rendering.

        Args:
            messages: pydantic-ai message list for this run.
        """
        self._replay_messages = messages

    def _resolve_display_registry(self) -> Mapping[str, Tool] | None:
        if self._display_registry_provider is not None:
            return self._display_registry_provider()
        return self._display_registry

    def _renderer(self) -> ToolRenderer:
        registry = self._resolve_display_registry()
        if registry is None:
            registry = core_display_registry()
        return ToolRenderer(registry)

    async def _event_stream_handler(
        self, ctx: RunContext, event_stream: AsyncIterable[AgentStreamEvent]
    ) -> None:
        try:
            self._raise_if_cancelled()
            async for event in event_stream:
                self._raise_if_cancelled()
                messages = getattr(ctx, "messages", None)
                if isinstance(messages, list):
                    self.set_replay_messages(messages)
                await self.on_event(event, ctx)
                self._raise_if_cancelled()
        finally:
            self._reset_response_stream_state()

    async def on_event(
        self, event: AgentStreamEvent, ctx: RunContext | None = None
    ) -> None:
        """Dispatch one agent stream event.

        Args:
            event: pydantic-ai stream event.
            ctx: Optional run context for query-result hydration.
        """
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
        previous = self._take_replaced_stream(event.index)
        if isinstance(event.part, TextPart):
            self._start_stream(
                event.index, TextPart, event.part.content, replace=previous
            )
        elif isinstance(event.part, ThinkingPart):
            self._start_stream(
                event.index, ThinkingPart, event.part.content, replace=previous
            )
        elif isinstance(event.part, ToolCallPart):
            self._tool_call_names[event.index] = event.part.tool_name
            self._tool_call_ids[event.index] = event.part.tool_call_id
            if event.part.tool_name == "execute_sql":
                self._ensure_sql_stream(event.index, replace=previous)
            elif previous is not None:
                previous.discard()
            self._set_tool_call_args(event.index, event.part.args)
            self._maybe_start_sql_generation_status(event.part.tool_name)
        elif previous is not None:
            previous.discard()

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
                candidate
                for candidate, tool_call_id in self._tool_call_ids.items()
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
                    stream = self._sql_streams.get(index)
                    if stream is not None:
                        stream.close()
            elif index is not None:
                self._discard_sql_stream(index)
            if index is not None:
                self._clear_tool_call_state(index)
            return

        blocks = self._renderer().executing(event.part.tool_name, args)
        if blocks:
            self.surface.emit(*blocks)
        if index is not None:
            self._discard_sql_stream(index)
            self._clear_tool_call_state(index)
        if event.part.tool_name == "viz":
            self.surface.status("Generating visualization...")
        elif event.part.tool_name == "analyze_data":
            self.surface.status("Analyzing data...")

    async def _on_tool_result(
        self, event: FunctionToolResultEvent, ctx: RunContext | None
    ) -> None:
        tool_name = event.part.tool_name
        if tool_name is None:
            self.surface.status(None)
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
        blocks = list(
            self._renderer().result(
                tool_name,
                content,
                context=ToolRenderContext(
                    tool_call_id=event.part.tool_call_id,
                    metadata=getattr(event.part, "metadata", None),
                    replay_messages=self._replay_messages,
                ),
            )
        )
        if complete_unavailable:
            blocks.append(md("*Complete result unavailable; showing preview.*"))
        if blocks:
            self.surface.emit(*blocks)
        self.surface.status("Crunching data...")

    def _start_stream(
        self,
        index: int,
        kind: type,
        text: str,
        *,
        replace: TextStream | None = None,
    ) -> None:
        self._finished_stream_indexes.discard(index)
        role: Role | None = "muted" if kind is ThinkingPart else None
        stream = self.surface.stream(role=role, replace=replace)
        self._text_streams[index] = stream
        self._stream_kinds[index] = kind
        if text:
            stream.append(text)

    def _append_stream(self, index: int, kind: type, text: str) -> None:
        if not text:
            return
        stream = self._text_streams.get(index)
        if stream is None or self._stream_kinds.get(index) is not kind:
            self._start_stream(index, kind, "")
            stream = self._text_streams[index]
        stream.append(text)

    def _take_replaced_stream(self, index: int) -> TextStream | None:
        response = self._text_streams.pop(index, None)
        sql = self._sql_streams.pop(index, None)
        self._stream_kinds.pop(index, None)
        self._finished_stream_indexes.discard(index)
        self._tool_call_names.pop(index, None)
        self._tool_call_args.pop(index, None)
        self._tool_call_ids.pop(index, None)
        self._sql_stream_queries.pop(index, None)
        if response is not None and sql is not None and response is not sql:
            sql.discard()
        return response or sql

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
        stream = self._ensure_sql_stream(index)
        stream.set(markdown_source(code(safe_query, "sql")))
        self._sql_stream_queries[index] = safe_query

    def _append_complete_sql(self, query: str) -> None:
        safe_query = sanitize_terminal_text(query)
        self.surface.emit(code(safe_query, "sql"))

    def _ensure_sql_stream(
        self, index: int, *, replace: TextStream | None = None
    ) -> TextStream:
        existing = self._sql_streams.get(index)
        if existing is not None:
            return existing
        later = sorted(
            candidate for candidate in self._sql_streams if candidate > index
        )
        before = self._sql_streams[later[0]] if later else None
        stream = self.surface.stream(replace=replace, before=before)
        self._sql_streams[index] = stream
        return stream

    def _clear_tool_call_state(self, index: int) -> None:
        self._tool_call_names.pop(index, None)
        self._tool_call_args.pop(index, None)
        self._tool_call_ids.pop(index, None)
        self._sql_streams.pop(index, None)
        self._sql_stream_queries.pop(index, None)

    def _discard_sql_stream(self, index: int) -> None:
        stream = self._sql_streams.get(index)
        if stream is not None:
            stream.discard()

    def _reset_tool_call_state(self, *, remove_previews: bool = False) -> None:
        if remove_previews:
            for stream in list(self._sql_streams.values()):
                stream.discard()
        self._tool_call_names.clear()
        self._tool_call_args.clear()
        self._tool_call_ids.clear()
        self._sql_streams.clear()
        self._sql_stream_queries.clear()

    def _finish_stream_segment(self, index: int) -> None:
        stream = self._text_streams.get(index)
        if stream is not None and index not in self._finished_stream_indexes:
            stream.close()
            self._finished_stream_indexes.add(index)

    def _finish_all_stream_segments(self) -> None:
        for index in list(self._text_streams):
            self._finish_stream_segment(index)

    def _reset_response_stream_state(self) -> None:
        self._text_streams.clear()
        self._stream_kinds.clear()
        self._finished_stream_indexes.clear()

    def _maybe_start_sql_generation_status(self, tool_name: str) -> None:
        if tool_name == "execute_sql":
            self.surface.status("Generating SQL...")

    def _announce_interrupted(self) -> None:
        app = getattr(self.surface, "app", None)
        append = getattr(app, "append_system_message", None)
        if callable(append):
            append("Query interrupted")
            return
        self.surface.emit(warn("Query interrupted"))

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
        """Run a query and present its stream.

        Args:
            user_query: User text.
            run_query: Agent entry that accepts ``event_stream_handler``.
            cancellation_token: Optional cooperative cancel event.
            message_history: Optional prior messages.

        Returns:
            The agent run, or None when interrupted.
        """
        self._reset_tool_call_state(remove_previews=True)
        self._finish_all_stream_segments()
        self._cancellation_token = cancellation_token
        try:
            self.log.info("streaming.execute.start")
            self.surface.status("Crunching data...")
            run = await run_query(
                user_query,
                message_history=message_history,
                event_stream_handler=self._event_stream_handler,
            )
            self.log.info("streaming.execute.end")
            return run
        except _QueryInterrupted:
            self._announce_interrupted()
            self.log.info("streaming.execute.interrupted")
            return None
        except asyncio.CancelledError:
            self._announce_interrupted()
            self.log.info("streaming.execute.cancelled")
            return None
        finally:
            self.surface.status(None)
            self._reset_tool_call_state(remove_previews=True)
            self._reset_response_stream_state()
            self._cancellation_token = None


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
