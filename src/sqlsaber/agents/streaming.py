"""Streaming utilities for agents."""

from typing import Any, Dict, List


class StreamingResponse:
    """Helper class to manage streaming response construction."""

    def __init__(self, content: List[Dict[str, Any]], stop_reason: str):
        self.content = content
        self.stop_reason = stop_reason


def build_tool_result_block(tool_use_id: str, content: str) -> Dict[str, Any]:
    """Build a tool result block for the conversation."""
    return {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}


def extract_sql_from_text(text: str) -> str:
    """Extract SQL query from markdown-formatted text."""
    if "```sql" in text:
        sql_start = text.find("```sql") + 6
        sql_end = text.find("```", sql_start)
        if sql_end > sql_start:
            return text[sql_start:sql_end].strip()
    return ""
