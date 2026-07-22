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
) -> tuple[dict[str, str], set[str]]:
    """Preload canonical JSON by tool-call ID without mutating message history."""
    hydrated: dict[str, str] = {}
    unavailable: set[str] = set()
    for reference in query_result_references_from_messages(messages):
        try:
            resolved = await resolve_query_result(
                reference,
                store=store,
                context=QueryResultContext(),
            )
        except QueryResultUnavailable:
            unavailable.add(reference.tool_call_id)
            continue
        try:
            hydrated[reference.tool_call_id] = resolved.data.decode("utf-8")
        except UnicodeDecodeError:
            unavailable.add(reference.tool_call_id)
    return hydrated, unavailable
