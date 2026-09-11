"""Wire vocabulary for ``saber rpc``.

Stdlib JSON only. Nothing here starts a query; nothing outside this module
builds or reads wire dicts.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Literal

from sqlsaber.artifacts import StoredArtifact
from sqlsaber.config.settings import ThinkingLevel
from sqlsaber.query_results import LoadedQueryResult, StoredQueryResult
from sqlsaber.utils.json_utils import EnhancedJSONEncoder

PROTOCOL_VERSION = 1
DEFAULT_RESULT_LIMIT = 500
MAX_RESULT_LIMIT = 5000
MAX_STDIN_LINE = 1_048_576
OVERSIZE_LINE = b"\x00OVERSIZE\n"

type RequestId = str | int
type Json = dict[str, Any]

THINKING_CHOICES: tuple[str, ...] = ("off", *(level.value for level in ThinkingLevel))

_UNSUPPORTED_FIELDS: dict[str, str] = {
    "streamingBehavior": (
        "streamingBehavior is not supported: SQLSaber has no steer/follow-up "
        "queue. Wait for agent_end, then send the prompt."
    ),
    "images": "images is not supported: SQLSaber prompts are text.",
    "parentSession": (
        "parentSession is not supported: SQLSaber threads are linear, "
        "not a session tree."
    ),
}


# --- commands (stdin) --------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Prompt:
    message: str
    id: RequestId | None = None


@dataclass(frozen=True, slots=True)
class Abort:
    id: RequestId | None = None


@dataclass(frozen=True, slots=True)
class NewSession:
    id: RequestId | None = None


@dataclass(frozen=True, slots=True)
class GetState:
    id: RequestId | None = None


@dataclass(frozen=True, slots=True)
class GetMessages:
    id: RequestId | None = None


@dataclass(frozen=True, slots=True)
class GetLastAssistantText:
    id: RequestId | None = None


@dataclass(frozen=True, slots=True)
class SetThinkingLevel:
    level: str
    id: RequestId | None = None


@dataclass(frozen=True, slots=True)
class ReloadModel:
    id: RequestId | None = None


@dataclass(frozen=True, slots=True)
class GetTables:
    id: RequestId | None = None


@dataclass(frozen=True, slots=True)
class GetQueryResult:
    result_id: str
    offset: int = 0
    limit: int = DEFAULT_RESULT_LIMIT
    id: RequestId | None = None


@dataclass(frozen=True, slots=True)
class GetArtifact:
    artifact_id: str
    id: RequestId | None = None


@dataclass(frozen=True, slots=True)
class Shutdown:
    id: RequestId | None = None


@dataclass(frozen=True, slots=True)
class Invalid:
    """A line the boundary refused. Never reaches the SDK."""

    error: str
    command: str = "parse"
    id: RequestId | None = None


type ReadCommand = (
    GetState | GetMessages | GetLastAssistantText | GetQueryResult | GetArtifact
)
"""Safe while a query runs: they only read SDK state or stores."""

type MutateCommand = NewSession | SetThinkingLevel | ReloadModel
"""Idle-only mutations. ``get_tables`` is idle-only too (live connections)."""

type Command = (
    Prompt | Abort | Shutdown | ReadCommand | MutateCommand | GetTables | Invalid
)


def parse_command(line: bytes) -> Command:
    """Decode one stdin record into a command. Total: never raises.

    Args:
        line: One JSONL record, optionally ``\\n``/``\\r\\n`` terminated.

    Returns:
        A typed command, or ``Invalid`` for any framing/JSON/field problem.
    """
    try:
        text = _decode_line(line)
    except UnicodeDecodeError:
        return Invalid("Command is not valid UTF-8")
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        return Invalid(f"Failed to parse command: {exc}")
    if not isinstance(obj, dict):
        return Invalid('Command must be a JSON object with a string "type"')
    raw_type = obj.get("type")
    if not isinstance(raw_type, str) or not raw_type:
        return Invalid('Command must be a JSON object with a string "type"')
    rid, id_error = _parse_id(obj.get("id"), raw_type, "id" in obj)
    if id_error is not None:
        return id_error
    for field, message in _UNSUPPORTED_FIELDS.items():
        if field in obj:
            return Invalid(message, command=raw_type, id=rid)
    match raw_type:
        case "prompt":
            return _parse_prompt(obj, rid)
        case "abort":
            return Abort(id=rid)
        case "new_session":
            return NewSession(id=rid)
        case "get_state":
            return GetState(id=rid)
        case "get_messages":
            return GetMessages(id=rid)
        case "get_last_assistant_text":
            return GetLastAssistantText(id=rid)
        case "set_thinking_level":
            return _parse_set_thinking_level(obj, rid)
        case "reload_model":
            return ReloadModel(id=rid)
        case "get_tables":
            return GetTables(id=rid)
        case "get_query_result":
            return _parse_get_query_result(obj, rid)
        case "get_artifact":
            return _parse_get_artifact(obj, rid)
        case "shutdown":
            return Shutdown(id=rid)
        case _:
            return Invalid(f"Unknown command: {raw_type}", command=raw_type, id=rid)


def _decode_line(line: bytes) -> str:
    text = line.decode("utf-8")
    if text.endswith("\n"):
        text = text[:-1]
    if text.endswith("\r"):
        text = text[:-1]
    return text


def _parse_id(
    value: object, command: str, present: bool
) -> tuple[RequestId | None, Invalid | None]:
    if not present or value is None:
        return None, None
    if isinstance(value, str):
        return value, None
    if isinstance(value, int) and not isinstance(value, bool):
        return value, None
    return None, Invalid("id must be a string or integer", command=command, id=None)


def _parse_prompt(obj: dict[str, Any], rid: RequestId | None) -> Command:
    message = obj.get("message")
    if not isinstance(message, str) or not message.strip():
        return Invalid("message must be a non-empty string", command="prompt", id=rid)
    return Prompt(message=message, id=rid)


def _parse_set_thinking_level(obj: dict[str, Any], rid: RequestId | None) -> Command:
    level = obj.get("level")
    if not isinstance(level, str) or level not in THINKING_CHOICES:
        choices = ", ".join(THINKING_CHOICES)
        return Invalid(
            f"level must be one of: {choices}",
            command="set_thinking_level",
            id=rid,
        )
    return SetThinkingLevel(level=level, id=rid)


def _parse_get_query_result(obj: dict[str, Any], rid: RequestId | None) -> Command:
    result_id = obj.get("resultId")
    if not isinstance(result_id, str) or not result_id:
        return Invalid(
            "resultId must be a non-empty string",
            command="get_query_result",
            id=rid,
        )
    offset = obj.get("offset", 0)
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        return Invalid(
            "offset must be a non-negative integer",
            command="get_query_result",
            id=rid,
        )
    limit = obj.get("limit", DEFAULT_RESULT_LIMIT)
    if (
        not isinstance(limit, int)
        or isinstance(limit, bool)
        or limit < 1
        or limit > MAX_RESULT_LIMIT
    ):
        return Invalid(
            f"limit must be an integer from 1 to {MAX_RESULT_LIMIT}",
            command="get_query_result",
            id=rid,
        )
    return GetQueryResult(result_id=result_id, offset=offset, limit=limit, id=rid)


def _parse_get_artifact(obj: dict[str, Any], rid: RequestId | None) -> Command:
    artifact_id = obj.get("artifactId")
    if not isinstance(artifact_id, str) or not artifact_id:
        return Invalid(
            "artifactId must be a non-empty string",
            command="get_artifact",
            id=rid,
        )
    return GetArtifact(artifact_id=artifact_id, id=rid)


# --- responses (stdout) ------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Ok:
    command: str
    id: RequestId | None
    data: Json | None = None


@dataclass(frozen=True, slots=True)
class Err:
    command: str
    id: RequestId | None
    error: str


type Response = Ok | Err


# --- transcript model --------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TextBlock:
    text: str


@dataclass(frozen=True, slots=True)
class ThinkingBlock:
    thinking: str


@dataclass(frozen=True, slots=True)
class ToolCallBlock:
    id: str
    name: str
    arguments: Json


type ContentBlock = TextBlock | ThinkingBlock | ToolCallBlock

type StopReason = Literal["stop", "length", "toolUse", "contentFilter", "error"]


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int


@dataclass(frozen=True, slots=True)
class RunUsageSummary(Usage):
    requests: int
    tool_calls: int
    context_tokens: int


@dataclass(frozen=True, slots=True)
class UserMessage:
    content: str
    timestamp_ms: int


@dataclass(frozen=True, slots=True)
class AssistantMessage:
    content: tuple[ContentBlock, ...]
    model: str | None
    usage: Usage | None
    stop_reason: StopReason | None
    timestamp_ms: int


@dataclass(frozen=True, slots=True)
class ToolResultMessage:
    tool_call_id: str
    tool_name: str
    content: Any
    is_error: bool
    timestamp_ms: int
    query_result: StoredQueryResult | None = None


type TranscriptMessage = UserMessage | AssistantMessage | ToolResultMessage


# --- state -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StateSnapshot:
    """Payload of ``ready``, ``get_state`` and mutating responses."""

    state: Literal["idle", "running"]
    database_names: tuple[str, ...]
    primary_database: str
    database_type: str
    model_name: str
    model_id: str | None
    thinking: str
    dangerous_mode: bool
    csv_tool_results: bool
    thread_id: str | None
    thread_persistence: bool
    message_count: int


@dataclass(frozen=True, slots=True)
class TableEntry:
    database: str
    schema: str
    name: str
    kind: str
    qualified_name: str


# --- events ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Ready:
    state: StateSnapshot
    protocol_version: int = PROTOCOL_VERSION


@dataclass(frozen=True, slots=True)
class AgentStart:
    prompt_id: RequestId | None = None


@dataclass(frozen=True, slots=True)
class MessageStart:
    pass


@dataclass(frozen=True, slots=True)
class TextStart:
    content_index: int


@dataclass(frozen=True, slots=True)
class TextDelta:
    content_index: int
    delta: str


@dataclass(frozen=True, slots=True)
class TextEnd:
    content_index: int
    content: str


@dataclass(frozen=True, slots=True)
class ThinkingStart:
    content_index: int


@dataclass(frozen=True, slots=True)
class ThinkingDelta:
    content_index: int
    delta: str


@dataclass(frozen=True, slots=True)
class ThinkingEnd:
    content_index: int
    content: str


@dataclass(frozen=True, slots=True)
class ToolCallStart:
    content_index: int
    id: str
    tool_name: str


@dataclass(frozen=True, slots=True)
class ToolCallDelta:
    content_index: int
    delta: str


@dataclass(frozen=True, slots=True)
class ToolCallEnd:
    content_index: int
    tool_call: ToolCallBlock


type ContentEvent = (
    TextStart
    | TextDelta
    | TextEnd
    | ThinkingStart
    | ThinkingDelta
    | ThinkingEnd
    | ToolCallStart
    | ToolCallDelta
    | ToolCallEnd
)


@dataclass(frozen=True, slots=True)
class MessageUpdate:
    event: ContentEvent


@dataclass(frozen=True, slots=True)
class MessageEnd:
    message: AssistantMessage


@dataclass(frozen=True, slots=True)
class SqlUpdate:
    sql: str
    tool_call_id: str | None = None


@dataclass(frozen=True, slots=True)
class ToolExecutionStart:
    tool_call_id: str
    tool_name: str
    args: Json


@dataclass(frozen=True, slots=True)
class ToolExecutionEnd:
    tool_call_id: str
    tool_name: str
    result: Any
    is_error: bool
    query_result: StoredQueryResult | None = None


@dataclass(frozen=True, slots=True)
class Completed:
    messages: tuple[TranscriptMessage, ...]
    text: str
    usage: RunUsageSummary | None
    query_results: tuple[StoredQueryResult, ...]
    artifacts: tuple[StoredArtifact, ...]
    thread_id: str | None


@dataclass(frozen=True, slots=True)
class Aborted:
    pass


@dataclass(frozen=True, slots=True)
class Failed:
    error: str


type RunOutcome = Completed | Aborted | Failed


@dataclass(frozen=True, slots=True)
class AgentEnd:
    outcome: RunOutcome


type Event = (
    Ready
    | AgentStart
    | MessageStart
    | MessageUpdate
    | MessageEnd
    | SqlUpdate
    | ToolExecutionStart
    | ToolExecutionEnd
    | AgentEnd
)


def encode(message: Response | Event) -> bytes:
    """One JSONL record (``\\n``-terminated, UTF-8, camelCase).

    Args:
        message: A response or event to serialize.

    Returns:
        UTF-8 bytes including a trailing newline.
    """
    return _dump(_wire_dict(message))


def _wire_dict(message: Response | Event) -> Json:
    match message:
        case Ok(command=command, id=rid, data=data):
            payload: Json = {
                "type": "response",
                "command": command,
                "success": True,
            }
            if data is not None:
                payload["data"] = data
            _put_id(payload, rid)
            return payload
        case Err(command=command, id=rid, error=error):
            payload = {
                "type": "response",
                "command": command,
                "success": False,
                "error": error,
            }
            _put_id(payload, rid)
            return payload
        case Ready(state=state, protocol_version=version):
            return {"type": "ready", "protocolVersion": version, **state_json(state)}
        case AgentStart(prompt_id=prompt_id):
            payload = {"type": "agent_start"}
            if prompt_id is not None:
                payload["promptId"] = prompt_id
            return payload
        case MessageStart():
            return {"type": "message_start"}
        case MessageUpdate(event=event):
            return {
                "type": "message_update",
                "assistantMessageEvent": _content_event_json(event),
            }
        case MessageEnd(message=transcript_message):
            return {"type": "message_end", "message": message_json(transcript_message)}
        case SqlUpdate(sql=sql, tool_call_id=tool_call_id):
            payload = {"type": "sql_update", "sql": sql}
            if tool_call_id:
                payload["toolCallId"] = tool_call_id
            return payload
        case ToolExecutionStart(tool_call_id=call_id, tool_name=name, args=args):
            return {
                "type": "tool_execution_start",
                "toolCallId": call_id,
                "toolName": name,
                "args": args,
            }
        case ToolExecutionEnd(
            tool_call_id=call_id,
            tool_name=name,
            result=result,
            is_error=is_error,
            query_result=query_result,
        ):
            payload = {
                "type": "tool_execution_end",
                "toolCallId": call_id,
                "toolName": name,
                "result": result,
                "isError": is_error,
            }
            if query_result is not None:
                payload["queryResult"] = query_result_json(query_result)
            return payload
        case AgentEnd(outcome=Completed() as completed):
            return {
                "type": "agent_end",
                "status": "completed",
                "text": completed.text,
                "messages": [message_json(item) for item in completed.messages],
                "usage": (
                    None
                    if completed.usage is None
                    else _run_usage_json(completed.usage)
                ),
                "queryResults": [
                    query_result_json(item) for item in completed.query_results
                ],
                "artifacts": [artifact_json(item) for item in completed.artifacts],
                "threadId": completed.thread_id,
            }
        case AgentEnd(outcome=Aborted()):
            return {"type": "agent_end", "status": "aborted"}
        case AgentEnd(outcome=Failed(error=error)):
            return {"type": "agent_end", "status": "error", "error": error}
    raise TypeError(f"unencodable RPC message: {type(message)!r}")


def state_json(state: StateSnapshot) -> Json:
    """Shared by ``Ready`` and every response carrying a snapshot.

    Args:
        state: Derived session snapshot.

    Returns:
        CamelCase wire object (without ``type`` / ``protocolVersion``).
    """
    return {
        "state": state.state,
        "database": {
            "name": state.primary_database,
            "type": state.database_type,
            "names": list(state.database_names),
        },
        "model": {"name": state.model_name, "id": state.model_id},
        "thinkingLevel": state.thinking,
        "thinkingLevels": list(THINKING_CHOICES),
        "dangerousMode": state.dangerous_mode,
        "csvToolResults": state.csv_tool_results,
        "threadId": state.thread_id,
        "threadPersistence": state.thread_persistence,
        "messageCount": state.message_count,
    }


def message_json(message: TranscriptMessage) -> Json:
    """Project one transcript message onto the wire.

    Args:
        message: A user, assistant, or tool-result message.

    Returns:
        CamelCase wire object.
    """
    match message:
        case UserMessage(content=content, timestamp_ms=timestamp_ms):
            return {
                "role": "user",
                "content": content,
                "timestamp": timestamp_ms,
            }
        case AssistantMessage(
            content=content,
            model=model,
            usage=usage,
            stop_reason=stop_reason,
            timestamp_ms=timestamp_ms,
        ):
            return {
                "role": "assistant",
                "content": [_block_json(block) for block in content],
                "model": model,
                "usage": None if usage is None else _usage_json(usage),
                "stopReason": stop_reason,
                "timestamp": timestamp_ms,
            }
        case ToolResultMessage(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            content=content,
            is_error=is_error,
            timestamp_ms=timestamp_ms,
            query_result=query_result,
        ):
            payload: Json = {
                "role": "toolResult",
                "toolCallId": tool_call_id,
                "toolName": tool_name,
                "content": content,
                "isError": is_error,
                "timestamp": timestamp_ms,
            }
            if query_result is not None:
                payload["queryResult"] = query_result_json(query_result)
            return payload
    raise TypeError(f"unencodable transcript message: {type(message)!r}")


def query_result_json(result: StoredQueryResult) -> Json:
    """CamelCase projection of a stored query-result descriptor.

    Args:
        result: Durable result handle.

    Returns:
        Wire object for ``queryResult`` fields.
    """
    payload: Json = {
        "id": result.id,
        "file": result.file,
        "rowCount": result.row_count,
        "columns": list(result.columns),
        "size": result.size,
        "sha256": result.sha256,
        "mediaType": result.media_type,
    }
    if result.database_name is not None:
        payload["databaseName"] = result.database_name
    if result.created_at is not None:
        payload["createdAt"] = result.created_at
    return payload


def artifact_json(artifact: StoredArtifact) -> Json:
    """CamelCase projection of a stored artifact descriptor.

    Args:
        artifact: Durable artifact handle.

    Returns:
        Wire object including ``uri`` (no inline bytes).
    """
    return {
        "id": artifact.id,
        "name": artifact.name,
        "kind": artifact.kind,
        "mediaType": artifact.media_type,
        "size": artifact.size,
        "sha256": artifact.sha256,
        "uri": artifact.uri,
    }


def query_result_page(loaded: LoadedQueryResult, *, offset: int, limit: int) -> Json:
    """Slice stored rows into a bounded wire page.

    Args:
        loaded: Complete stored result.
        offset: Starting row index.
        limit: Maximum rows to return.

    Returns:
        ``result``, ``offset``, ``limit``, ``rows``, ``hasMore``.
    """
    rows = loaded.rows()
    page = rows[offset : offset + limit]
    return {
        "result": query_result_json(loaded.descriptor),
        "offset": offset,
        "limit": limit,
        "rows": page,
        "hasMore": offset + len(page) < len(rows),
    }


def _content_event_json(event: ContentEvent) -> Json:
    match event:
        case TextStart(content_index=index):
            return {"type": "text_start", "contentIndex": index}
        case TextDelta(content_index=index, delta=delta):
            return {"type": "text_delta", "contentIndex": index, "delta": delta}
        case TextEnd(content_index=index, content=content):
            return {"type": "text_end", "contentIndex": index, "content": content}
        case ThinkingStart(content_index=index):
            return {"type": "thinking_start", "contentIndex": index}
        case ThinkingDelta(content_index=index, delta=delta):
            return {
                "type": "thinking_delta",
                "contentIndex": index,
                "delta": delta,
            }
        case ThinkingEnd(content_index=index, content=content):
            return {
                "type": "thinking_end",
                "contentIndex": index,
                "content": content,
            }
        case ToolCallStart(content_index=index, id=call_id, tool_name=name):
            return {
                "type": "toolcall_start",
                "contentIndex": index,
                "id": call_id,
                "toolName": name,
            }
        case ToolCallDelta(content_index=index, delta=delta):
            return {"type": "toolcall_delta", "contentIndex": index, "delta": delta}
        case ToolCallEnd(content_index=index, tool_call=tool_call):
            return {
                "type": "toolcall_end",
                "contentIndex": index,
                "toolCall": _block_json(tool_call),
            }
    raise TypeError(f"unencodable content event: {type(event)!r}")


def _block_json(block: ContentBlock) -> Json:
    match block:
        case TextBlock(text=text):
            return {"type": "text", "text": text}
        case ThinkingBlock(thinking=thinking):
            return {"type": "thinking", "thinking": thinking}
        case ToolCallBlock(id=call_id, name=name, arguments=arguments):
            return {
                "type": "toolCall",
                "id": call_id,
                "name": name,
                "arguments": arguments,
            }
    raise TypeError(f"unencodable content block: {type(block)!r}")


def _usage_json(usage: Usage) -> Json:
    return {
        "inputTokens": usage.input_tokens,
        "outputTokens": usage.output_tokens,
        "cacheReadTokens": usage.cache_read_tokens,
        "cacheWriteTokens": usage.cache_write_tokens,
    }


def _run_usage_json(usage: RunUsageSummary) -> Json:
    payload = _usage_json(usage)
    payload["requests"] = usage.requests
    payload["toolCalls"] = usage.tool_calls
    payload["contextTokens"] = usage.context_tokens
    return payload


def _put_id(payload: Json, rid: RequestId | None) -> None:
    if rid is not None:
        payload["id"] = rid


def _sanitize(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    return value


def _dump(payload: Json) -> bytes:
    text = json.dumps(
        _sanitize(payload),
        ensure_ascii=False,
        allow_nan=False,
        cls=EnhancedJSONEncoder,
    )
    return (text + "\n").encode("utf-8")
