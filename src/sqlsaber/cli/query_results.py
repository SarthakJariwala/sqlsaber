"""CLI query-result store construction and asynchronous hydration helpers."""

from __future__ import annotations

from pathlib import Path

import platformdirs
from pydantic_ai.messages import ModelMessage

from sqlsaber.query_result_resolution import (
    query_result_references_from_messages,
    resolve_query_result,
)
from sqlsaber.query_results import (
    FilesystemQueryResultStore,
    QueryResultContext,
    QueryResultStore,
    QueryResultUnavailable,
)


def cli_query_result_store() -> FilesystemQueryResultStore:
    """Return the persistent store used by all CLI execution/replay paths."""
    return FilesystemQueryResultStore(
        Path(platformdirs.user_data_dir("sqlsaber")) / "query-results"
    )


async def hydrate_query_result_contents(
    messages: list[ModelMessage],
    *,
    store: QueryResultStore,
) -> tuple[dict[tuple[str, str], str], set[tuple[str, str]]]:
    """Preload canonical JSON by tool name and call ID without mutating history."""
    hydrated: dict[tuple[str, str], str] = {}
    unavailable: set[tuple[str, str]] = set()
    for reference in query_result_references_from_messages(messages):
        key = (reference.tool_name, reference.tool_call_id)
        try:
            resolved = await resolve_query_result(
                reference,
                store=store,
                context=QueryResultContext(),
            )
        except QueryResultUnavailable:
            unavailable.add(key)
            continue
        try:
            hydrated[key] = resolved.data.decode("utf-8")
        except UnicodeDecodeError:
            unavailable.add(key)
    return hydrated, unavailable
