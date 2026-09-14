"""Partial JSON helpers shared by the CLI stream presenter and RPC."""

from __future__ import annotations

from pydantic_core import from_json


def partial_json_query(args: str) -> str | None:
    """Decode the complete portion of a query value from partial JSON arguments.

    Args:
        args: A possibly incomplete JSON object string, typically streamed
            ``execute_sql`` tool-call arguments.

    Returns:
        The ``query`` string when it can be recovered, otherwise ``None``.
    """
    try:
        parsed = from_json(args, allow_partial="trailing-strings")
    except ValueError:
        return None
    if not isinstance(parsed, dict):
        return None
    query = parsed.get("query")
    return query if isinstance(query, str) else None
