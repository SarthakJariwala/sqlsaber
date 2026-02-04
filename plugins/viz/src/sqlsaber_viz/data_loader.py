"""Helpers for loading SQL result payloads and extracting summaries."""

from __future__ import annotations

import json
from datetime import date, datetime, time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic_ai import RunContext


def find_tool_output_payload(ctx: "RunContext", tool_call_id: str) -> dict | None:
    """Find tool output from RunContext message history."""
    return find_tool_output_in_messages(ctx.messages, tool_call_id)


def find_tool_output_in_messages(messages: list, tool_call_id: str) -> dict | None:
    """Find tool output from a list of ModelMessage objects."""
    for message in reversed(messages):
        for part in getattr(message, "parts", []):
            if getattr(part, "part_kind", "") not in (
                "tool-return",
                "builtin-tool-return",
            ):
                continue
            if getattr(part, "tool_call_id", None) != tool_call_id:
                continue
            content = getattr(part, "content", None)
            if isinstance(content, dict):
                return content
            if isinstance(content, str):
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError:
                    return {"result": content}
                if isinstance(parsed, dict):
                    return parsed
                return {"result": parsed}
    return None


def extract_data_summary(payload: dict) -> dict:
    """Extract column info and samples from SQL result payload.

    Returns:
        {
            "columns": [
                {"name": "col1", "type": "string", "sample": ["a", "b", "c"]},
                {"name": "col2", "type": "number", "sample": [1, 2, 3]},
            ],
            "row_count": 150,
            "rows": [...]  # Full rows for rendering
        }
    """

    results = payload.get("results")
    rows = _coerce_rows(results) if isinstance(results, list) else []
    row_count = payload.get("row_count")
    if not isinstance(row_count, int):
        row_count = len(rows)

    columns = _extract_columns(rows)
    return {"columns": columns, "row_count": row_count, "rows": rows}


def infer_column_type(values: list[object]) -> str:
    """Infer column type from sample values.

    Returns: "number", "string", "time", "boolean", or "null"
    """

    cleaned = [value for value in values if value is not None]
    if not cleaned:
        return "null"

    if all(isinstance(value, bool) for value in cleaned):
        return "boolean"

    if all(isinstance(value, (int, float)) for value in cleaned):
        return "number"

    if all(_is_time_value(value) for value in cleaned):
        return "time"

    return "string"


def _extract_columns(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    if not rows:
        return []

    # Use the union of keys from the first 50 rows to avoid missing sparse columns.
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows[:50]:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                keys.append(key)

    columns: list[dict[str, object]] = []
    for key in keys:
        sample_values = [row.get(key) for row in rows[:20] if key in row]
        column_type = infer_column_type(sample_values)
        columns.append(
            {
                "name": key,
                "type": column_type,
                "sample": sample_values[:5],
            }
        )

    return columns


def _coerce_rows(rows: list[object]) -> list[dict[str, object]]:
    coerced: list[dict[str, object]] = []
    for row in rows:
        if isinstance(row, dict):
            coerced.append({str(key): value for key, value in row.items()})
        else:
            coerced.append({"value": row})
    return coerced


def _is_time_value(value: object) -> bool:
    if isinstance(value, (datetime, date, time)):
        return True
    if isinstance(value, str):
        normalized = value
        if value.endswith("Z"):
            normalized = value[:-1] + "+00:00"
        try:
            datetime.fromisoformat(normalized)
            return True
        except ValueError:
            try:
                time.fromisoformat(normalized)
                return True
            except ValueError:
                return False
    return False
