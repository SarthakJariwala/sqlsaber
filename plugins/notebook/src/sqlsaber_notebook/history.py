"""Part-preserving notebook snapshot collapse for outbound model history."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from pydantic_ai.messages import (
    CachePoint,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    UserContent,
    UserPromptPart,
)

SNAPSHOT_MARKER = "Current notebook state:"
# Anthropic permits four effective breakpoints. Cached instructions and tool
# definitions consume two, leaving at most two explicit historical stubs.
_MAX_EXPLICIT_CACHE_POINTS = 2


async def collapse_old_snapshots(
    messages: list[ModelMessage],
    *,
    cache: bool,
) -> list[ModelMessage]:
    """Replace all but the newest full snapshot without touching tool returns."""

    snapshot_indices = [
        index for index, message in enumerate(messages) if _carries_snapshot(message)
    ]
    if len(snapshot_indices) <= 1:
        return messages
    old_indices = snapshot_indices[:-1]
    cache_indices = set(old_indices[-_MAX_EXPLICIT_CACHE_POINTS:]) if cache else set()
    old_set = set(old_indices)
    return [
        _collapse_request(message, cache=index in cache_indices)
        if index in old_set and isinstance(message, ModelRequest)
        else message
        for index, message in enumerate(messages)
    ]


def _collapse_request(request: ModelRequest, *, cache: bool) -> ModelRequest:
    snapshot_parts = [
        index
        for index, part in enumerate(request.parts)
        if isinstance(part, UserPromptPart)
        and _content_has_marker(part.content, SNAPSHOT_MARKER)
    ]
    cache_part = snapshot_parts[-1] if cache and snapshot_parts else None
    parts: list[ModelRequestPart] = []
    for index, part in enumerate(request.parts):
        if index in snapshot_parts and isinstance(part, UserPromptPart):
            content: list[UserContent] = ["[old notebook state omitted]"]
            if index == cache_part:
                content.append(CachePoint())
            parts.append(replace(part, content=content))
        else:
            parts.append(part)
    return replace(request, parts=parts)


def _carries_snapshot(message: ModelMessage) -> bool:
    return isinstance(message, ModelRequest) and any(
        isinstance(part, UserPromptPart)
        and _content_has_marker(part.content, SNAPSHOT_MARKER)
        for part in message.parts
    )


def _content_has_marker(content: str | Sequence[UserContent], marker: str) -> bool:
    if isinstance(content, str):
        return marker in content
    return any(isinstance(item, str) and marker in item for item in content)
