"""Streaming utilities for agents."""

from typing import Any


class StreamingResponse:
    """Helper class to manage streaming response construction."""

    def __init__(self, content: list[dict[str, Any]], stop_reason: str):
        self.content = content
        self.stop_reason = stop_reason


def build_tool_result_block(tool_use_id: str, content: str) -> dict[str, Any]:
    """Build a tool result block for the conversation."""
    return {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}
