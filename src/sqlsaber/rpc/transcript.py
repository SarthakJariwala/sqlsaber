"""Project pydantic-ai history and stream events onto the RPC wire model."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic_ai.messages import (
    AgentStreamEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    RetryPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
    ToolReturnPart,
    UserPromptPart,
)

from sqlsaber.query_result_resolution import query_result_from_metadata
from sqlsaber.utils.partial_json import partial_json_query

from .protocol import (
    AssistantMessage,
    ContentBlock,
    Event,
    MessageEnd,
    MessageStart,
    MessageUpdate,
    SqlUpdate,
    StopReason,
    TextBlock,
    TextDelta,
    TextEnd,
    TextStart,
    ThinkingBlock,
    ThinkingDelta,
    ThinkingEnd,
    ThinkingStart,
    ToolCallBlock,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
    ToolExecutionEnd,
    ToolExecutionStart,
    ToolResultMessage,
    TranscriptMessage,
    Usage,
    UserMessage,
)

_STOP_REASONS: dict[str, StopReason] = {
    "stop": "stop",
    "length": "length",
    "tool_call": "toolUse",
    "content_filter": "contentFilter",
    "error": "error",
}


def transcript(messages: Sequence[ModelMessage]) -> list[TranscriptMessage]:
    """Project committed pydantic-ai history onto the RPC transcript.

    Args:
        messages: SDK-owned ``ModelMessage`` history.

    Returns:
        Wire transcript messages in order. System-prompt parts are dropped.
    """
    projected: list[TranscriptMessage] = []
    for message in messages:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, UserPromptPart):
                    projected.append(user_message(part))
                elif isinstance(part, (ToolReturnPart, RetryPromptPart)):
                    projected.append(tool_result_message(part))
        elif isinstance(message, ModelResponse):
            projected.append(assistant_message(message))
    return projected


def last_assistant_text(messages: Sequence[ModelMessage]) -> str | None:
    """Return concatenated text of the latest assistant message, if any.

    Args:
        messages: SDK-owned history.

    Returns:
        Joined text blocks, an empty string when the last assistant message
        has no text, or ``None`` when there is no assistant message.
    """
    for message in reversed(transcript(messages)):
        if isinstance(message, AssistantMessage):
            return "".join(
                block.text for block in message.content if isinstance(block, TextBlock)
            )
    return None


def user_message(part: UserPromptPart) -> UserMessage:
    """Project a user-prompt part.

    Args:
        part: pydantic-ai user prompt.

    Returns:
        Wire user message.
    """
    return UserMessage(
        content=_user_text(part.content),
        timestamp_ms=_timestamp_ms(part.timestamp),
    )


def assistant_message(response: ModelResponse) -> AssistantMessage:
    """Project a model response.

    Args:
        response: pydantic-ai model response.

    Returns:
        Wire assistant message.
    """
    usage = response.usage
    return AssistantMessage(
        content=content_blocks(response.parts),
        model=response.model_name,
        usage=Usage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            cache_write_tokens=usage.cache_write_tokens,
        ),
        stop_reason=_STOP_REASONS.get(response.finish_reason or "", None),
        timestamp_ms=_timestamp_ms(response.timestamp),
    )


def tool_result_message(part: ToolReturnPart | RetryPromptPart) -> ToolResultMessage:
    """Project a tool return or retry prompt.

    Args:
        part: Tool result or retry feedback.

    Returns:
        Wire tool-result message.
    """
    if isinstance(part, RetryPromptPart):
        return ToolResultMessage(
            tool_call_id=part.tool_call_id,
            tool_name=part.tool_name or "",
            content=part.model_response(),
            is_error=True,
            timestamp_ms=_timestamp_ms(part.timestamp),
        )
    return ToolResultMessage(
        tool_call_id=part.tool_call_id,
        tool_name=part.tool_name,
        content=part.content,
        is_error=part.outcome != "success",
        timestamp_ms=_timestamp_ms(part.timestamp),
        query_result=query_result_from_metadata(part.metadata),
    )


def content_blocks(parts: Sequence[ModelResponsePart]) -> tuple[ContentBlock, ...]:
    """Map response parts onto transcript content blocks.

    Args:
        parts: Model response parts.

    Returns:
        Text, thinking, and tool-call blocks. File and builtin parts are dropped.
    """
    blocks: list[ContentBlock] = []
    for part in parts:
        if isinstance(part, TextPart):
            blocks.append(TextBlock(text=part.content))
        elif isinstance(part, ThinkingPart):
            blocks.append(ThinkingBlock(thinking=part.content))
        elif isinstance(part, ToolCallPart):
            blocks.append(
                ToolCallBlock(
                    id=part.tool_call_id,
                    name=part.tool_name,
                    arguments=part.args_as_dict(),
                )
            )
    return tuple(blocks)


def _user_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence) and not isinstance(content, (bytes, bytearray)):
        pieces: list[str] = []
        for item in content:
            if isinstance(item, str):
                pieces.append(item)
            else:
                text = getattr(item, "content", None)
                if isinstance(text, str):
                    pieces.append(text)
        return "".join(pieces)
    return str(content)


def _timestamp_ms(value: datetime | None) -> int:
    if value is None:
        return int(datetime.now(timezone.utc).timestamp() * 1000)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return int(value.timestamp() * 1000)


@dataclass(slots=True)
class _OpenBlock:
    kind: Literal["text", "thinking", "tool_call"]
    text: str = ""
    tool_call_id: str = ""
    tool_name: str = ""
    args: str | dict[str, Any] = ""
    last_sql: str = ""


class StreamTranslator:
    """Translate one ``event_stream_handler`` invocation into wire events.

    pydantic-ai calls the handler once per graph node, so one translator
    instance is at most one assistant message.
    """

    def __init__(self, *, model_name: str | None) -> None:
        self._model_name = model_name
        self._blocks: dict[int, _OpenBlock] = {}
        self._completed: dict[int, ContentBlock] = {}
        self._started = False

    def translate(self, event: AgentStreamEvent) -> list[Event]:
        """Return zero or more wire events for one pydantic-ai stream event.

        Args:
            event: A part-start/delta/end or function-tool event.

        Returns:
            Wire events in the order a client should observe them.
        """
        if isinstance(event, PartStartEvent):
            return self._on_part_start(event)
        if isinstance(event, PartDeltaEvent):
            return self._on_part_delta(event)
        if isinstance(event, PartEndEvent):
            return self._on_part_end(event)
        if isinstance(event, FunctionToolCallEvent):
            return [
                ToolExecutionStart(
                    tool_call_id=event.part.tool_call_id,
                    tool_name=event.part.tool_name,
                    args=event.part.args_as_dict(),
                )
            ]
        if isinstance(event, FunctionToolResultEvent):
            return [self._tool_result(event)]
        return []

    def finish(self) -> list[Event]:
        """Close any open message with ``message_end``.

        Returns:
            End events for unfinished blocks plus ``MessageEnd`` when a
            message was opened; otherwise an empty list.
        """
        events: list[Event] = []
        for index in sorted(self._blocks):
            events.extend(self._close_index(index))
        if self._started:
            events.append(
                MessageEnd(
                    message=AssistantMessage(
                        content=tuple(
                            self._completed[index] for index in sorted(self._completed)
                        ),
                        model=self._model_name,
                        usage=None,
                        stop_reason=None,
                        timestamp_ms=_timestamp_ms(None),
                    )
                )
            )
        return events

    def _ensure_message_start(self) -> list[Event]:
        if self._started:
            return []
        self._started = True
        return [MessageStart()]

    def _on_part_start(self, event: PartStartEvent) -> list[Event]:
        events = self._ensure_message_start()
        if event.index in self._blocks:
            events.extend(self._close_index(event.index))
        part = event.part
        if isinstance(part, TextPart):
            self._blocks[event.index] = _OpenBlock(kind="text", text=part.content)
            events.append(MessageUpdate(TextStart(event.index)))
            if part.content:
                events.append(MessageUpdate(TextDelta(event.index, part.content)))
            return events
        if isinstance(part, ThinkingPart):
            self._blocks[event.index] = _OpenBlock(kind="thinking", text=part.content)
            events.append(MessageUpdate(ThinkingStart(event.index)))
            if part.content:
                events.append(MessageUpdate(ThinkingDelta(event.index, part.content)))
            return events
        if isinstance(part, ToolCallPart):
            args: str | dict[str, Any]
            if isinstance(part.args, dict):
                args = dict(part.args)
            else:
                args = part.args or ""
            self._blocks[event.index] = _OpenBlock(
                kind="tool_call",
                tool_call_id=part.tool_call_id,
                tool_name=part.tool_name,
                args=args,
            )
            events.append(
                MessageUpdate(
                    ToolCallStart(event.index, part.tool_call_id, part.tool_name)
                )
            )
            delta = _args_delta(args)
            if delta:
                events.append(MessageUpdate(ToolCallDelta(event.index, delta)))
            events.extend(self._maybe_sql_update(event.index))
            return events
        return events

    def _on_part_delta(self, event: PartDeltaEvent) -> list[Event]:
        delta = event.delta
        block = self._blocks.get(event.index)
        if isinstance(delta, TextPartDelta):
            text = delta.content_delta or ""
            if block is None or block.kind != "text":
                events = self._ensure_message_start()
                if block is not None:
                    events.extend(self._close_index(event.index))
                self._blocks[event.index] = _OpenBlock(kind="text")
                events.append(MessageUpdate(TextStart(event.index)))
                block = self._blocks[event.index]
            else:
                events = []
            block.text += text
            if text:
                events.append(MessageUpdate(TextDelta(event.index, text)))
            return events
        if isinstance(delta, ThinkingPartDelta):
            text = delta.content_delta or ""
            if block is None or block.kind != "thinking":
                events = self._ensure_message_start()
                if block is not None:
                    events.extend(self._close_index(event.index))
                self._blocks[event.index] = _OpenBlock(kind="thinking")
                events.append(MessageUpdate(ThinkingStart(event.index)))
                block = self._blocks[event.index]
            else:
                events = []
            block.text += text
            if text:
                events.append(MessageUpdate(ThinkingDelta(event.index, text)))
            return events
        if isinstance(delta, ToolCallPartDelta):
            events = []
            if block is None or block.kind != "tool_call":
                events.extend(self._ensure_message_start())
                if block is not None:
                    events.extend(self._close_index(event.index))
                self._blocks[event.index] = _OpenBlock(kind="tool_call")
                block = self._blocks[event.index]
                events.append(
                    MessageUpdate(ToolCallStart(event.index, block.tool_call_id, ""))
                )
            if delta.tool_name_delta:
                block.tool_name = f"{block.tool_name}{delta.tool_name_delta}"
            if delta.tool_call_id:
                block.tool_call_id = delta.tool_call_id
            args_delta = delta.args_delta
            emitted = ""
            if isinstance(args_delta, str):
                if isinstance(block.args, str):
                    block.args = block.args + args_delta
                else:
                    block.args = json.dumps(block.args, ensure_ascii=False) + args_delta
                emitted = args_delta
            elif isinstance(args_delta, dict):
                if isinstance(block.args, dict):
                    block.args = {**block.args, **args_delta}
                else:
                    parsed = _parse_args_object(block.args)
                    block.args = {**parsed, **args_delta}
                emitted = json.dumps(args_delta, ensure_ascii=False)
            if emitted:
                events.append(MessageUpdate(ToolCallDelta(event.index, emitted)))
            events.extend(self._maybe_sql_update(event.index))
            return events
        return []

    def _on_part_end(self, event: PartEndEvent) -> list[Event]:
        part = event.part
        if isinstance(part, TextPart):
            self._blocks.pop(event.index, None)
            self._completed[event.index] = TextBlock(text=part.content)
            return [MessageUpdate(TextEnd(event.index, part.content))]
        if isinstance(part, ThinkingPart):
            self._blocks.pop(event.index, None)
            self._completed[event.index] = ThinkingBlock(thinking=part.content)
            return [MessageUpdate(ThinkingEnd(event.index, part.content))]
        if isinstance(part, ToolCallPart):
            self._blocks.pop(event.index, None)
            tool_call = ToolCallBlock(
                id=part.tool_call_id,
                name=part.tool_name,
                arguments=part.args_as_dict(),
            )
            self._completed[event.index] = tool_call
            events: list[Event] = [MessageUpdate(ToolCallEnd(event.index, tool_call))]
            query = tool_call.arguments.get("query")
            if part.tool_name == "execute_sql" and isinstance(query, str) and query:
                events.append(SqlUpdate(sql=query, tool_call_id=part.tool_call_id))
            return events
        return []

    def _close_index(self, index: int) -> list[Event]:
        block = self._blocks.pop(index, None)
        if block is None:
            return []
        if block.kind == "text":
            self._completed[index] = TextBlock(text=block.text)
            return [MessageUpdate(TextEnd(index, block.text))]
        if block.kind == "thinking":
            self._completed[index] = ThinkingBlock(thinking=block.text)
            return [MessageUpdate(ThinkingEnd(index, block.text))]
        arguments = _args_as_dict(block.args)
        tool_call = ToolCallBlock(
            id=block.tool_call_id,
            name=block.tool_name,
            arguments=arguments,
        )
        self._completed[index] = tool_call
        return [MessageUpdate(ToolCallEnd(index, tool_call))]

    def _maybe_sql_update(self, index: int) -> list[Event]:
        block = self._blocks.get(index)
        if (
            block is None
            or block.kind != "tool_call"
            or block.tool_name != "execute_sql"
        ):
            return []
        query: str | None
        if isinstance(block.args, str):
            query = partial_json_query(block.args)
        else:
            value = block.args.get("query")
            query = value if isinstance(value, str) else None
        if not query or query == block.last_sql:
            return []
        block.last_sql = query
        return [SqlUpdate(sql=query, tool_call_id=block.tool_call_id or None)]

    def _tool_result(self, event: FunctionToolResultEvent) -> ToolExecutionEnd:
        part = event.part
        if isinstance(part, RetryPromptPart):
            return ToolExecutionEnd(
                tool_call_id=part.tool_call_id,
                tool_name=part.tool_name or "",
                result=part.model_response(),
                is_error=True,
            )
        return ToolExecutionEnd(
            tool_call_id=part.tool_call_id,
            tool_name=part.tool_name,
            result=part.content,
            is_error=part.outcome != "success",
            query_result=query_result_from_metadata(getattr(part, "metadata", None)),
        )


def _args_delta(args: str | dict[str, Any]) -> str:
    if isinstance(args, str):
        return args
    if not args:
        return ""
    return json.dumps(args, ensure_ascii=False)


def _parse_args_object(raw: str) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _args_as_dict(args: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(args, dict):
        return args
    return _parse_args_object(args)
