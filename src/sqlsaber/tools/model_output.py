"""Compact model-facing SQL output; structured results remain available to clients."""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Awaitable, Callable, Mapping
from functools import wraps
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from pydantic_ai import ToolReturn

from sqlsaber.utils.json_utils import json_dumps


def _json(value: object) -> str:
    return json_dumps(value, ensure_ascii=False, separators=(",", ":"))


def _csv(label: str, rows: list[dict[str, Any]]) -> str:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\r\n")
    writer.writerow(columns)
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column)
            if value is None:
                cells.append("\\N")
            else:
                text = value if isinstance(value, str) else _json(value)
                cells.append(text.replace("\\", "\\\\"))
        writer.writerow(cells)
    return f"{label} (CSV; null=\\N; backslashes escaped):\n{output.getvalue()}"


def format_sql_output(name: str, payload: Mapping[str, Any]) -> str:
    """Use CSV for rows, JSON for metadata and nested cell values.

    CSV is a model presentation, not a replacement for typed JSON storage.
    Normalize database-specific values with the same encoder as retained results.
    """
    data = json.loads(_json(payload))
    if "error" in data:
        return _json(data)
    if name == "introspect_schema":
        sections = []
        for name, info in data.items():
            columns = info.pop("columns", {})
            sections.append(_json({"table": name, **info}))
            sections.append(
                _csv(
                    "columns",
                    [{"name": key, **value} for key, value in columns.items()],
                )
            )
        return "\n".join(sections) if sections else "{}"
    for key in ("tables", "databases", "results", "preview_rows"):
        if key in data:
            if not data[key]:
                return _json(data)
            rows = data.pop(key)
            rows = [row if isinstance(row, dict) else {"value": row} for row in rows]
            return _json(data) + "\n" + _csv(key, rows)
    return _json(data)


def model_output_wrapper(
    function: Callable[..., Awaitable[Any]], name: str
) -> Callable[..., Awaitable[ToolReturn]]:
    """Convert only the agent boundary, preserving signatures and UI metadata."""

    from pydantic_ai import ToolReturn

    @wraps(function)
    async def wrapped(*args: Any, **kwargs: Any) -> ToolReturn:
        result = await function(*args, **kwargs)
        returned = result if isinstance(result, ToolReturn) else ToolReturn(result)
        payload = json.loads(cast(str, returned.return_value))
        return ToolReturn(
            return_value=format_sql_output(name, payload),
            content=returned.content,
            metadata={
                **(returned.metadata or {}),
                "sqlsaber_structured_result": payload,
            },
        )

    return wrapped
