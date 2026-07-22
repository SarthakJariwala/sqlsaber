from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest
from pydantic_ai import ToolReturn
from pydantic_ai.messages import (
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
)

from sqlsaber.cli.query_result_gc import collect_cli_query_results

from sqlsaber.query_result_resolution import (
    find_query_result_reference,
    query_result_references_from_messages,
    resolve_query_result,
)
from sqlsaber.query_results import (
    MAX_MODEL_QUERY_RESULT_BYTES,
    FilesystemQueryResultStore,
    InMemoryQueryResultStore,
    QueryResultContext,
    QueryResultData,
    QueryResultUnavailable,
    build_model_projection,
    descriptor_for_data,
    new_query_result_id,
)
from sqlsaber.threads.storage import ThreadStorage
from sqlsaber.tools.sql_tools import ExecuteSQLTool


def _descriptor(data: bytes, *, file: str = "result_call.json"):
    return descriptor_for_data(
        data,
        result_id=new_query_result_id(),
        file=file,
        row_count=1,
        columns=("value",),
        database_name="test",
    )


@pytest.mark.asyncio
async def test_in_memory_and_filesystem_stores_round_trip_exact_bytes(tmp_path) -> None:
    data = b'{"success":true,"row_count":1,"results":[{"value":1}],"file":"result_call.json"}'
    context = QueryResultContext(conversation_id="conversation-1")

    memory = InMemoryQueryResultStore()
    descriptor = await memory.put(
        QueryResultData(data), descriptor=_descriptor(data), context=context
    )
    assert (await memory.get(descriptor.id, context=context)).data == data

    first = FilesystemQueryResultStore(tmp_path / "results")
    filesystem_descriptor = await first.put(
        QueryResultData(data), descriptor=_descriptor(data), context=context
    )
    second = FilesystemQueryResultStore(tmp_path / "results")
    loaded = await second.get(filesystem_descriptor.id, context=context)
    assert loaded.data == data
    assert loaded.descriptor == filesystem_descriptor


@pytest.mark.asyncio
async def test_filesystem_store_detects_corruption(tmp_path) -> None:
    data = b'{"success":true,"results":[{"value":1}]}'
    store = FilesystemQueryResultStore(tmp_path / "results")
    descriptor = await store.put(
        QueryResultData(data),
        descriptor=_descriptor(data),
        context=QueryResultContext(),
    )
    result_path = store.root / descriptor.id[3:5] / descriptor.id / "result.json"
    result_path.write_bytes(b"corrupt")

    with pytest.raises(QueryResultUnavailable):
        await store.get(descriptor.id, context=QueryResultContext())


def test_large_projection_is_deterministic_valid_and_bounded() -> None:
    rows = [{"value": "x" * 500, "position": index} for index in range(1000)]
    payload = {
        "success": True,
        "row_count": len(rows),
        "results": rows,
        "file": "result_call.json",
        "auto_limit_applied": True,
    }
    canonical = json.dumps(payload).encode()
    descriptor = descriptor_for_data(
        canonical,
        result_id="qr_0123456789abcdef0123456789abcdef",
        file="result_call.json",
        row_count=len(rows),
        columns=("value", "position"),
    )

    first = build_model_projection(payload, descriptor)
    second = build_model_projection(payload, descriptor)
    encoded = json.dumps(first, ensure_ascii=False, separators=(",", ":")).encode()

    assert first == second
    assert len(encoded) <= MAX_MODEL_QUERY_RESULT_BYTES
    assert first["results_truncated"] is True
    assert first["auto_limit_applied"] is True
    assert len(first["preview_rows"]) < len(rows)


@pytest.mark.asyncio
async def test_message_resolution_uses_descriptor_and_validates_store() -> None:
    payload = {
        "success": True,
        "row_count": 1,
        "results": [{"value": 1}],
        "file": "result_call.json",
    }
    data = json.dumps(payload).encode()
    store = InMemoryQueryResultStore()
    descriptor = await store.put(
        QueryResultData(data),
        descriptor=_descriptor(data),
        context=QueryResultContext(),
    )
    messages = [
        ModelResponse(
            parts=[ToolCallPart("execute_sql", {"query": "select 1"}, "call")]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    "execute_sql",
                    '{"success":true,"preview_rows":[]}',
                    "call",
                    metadata={"query_result": descriptor.to_dict()},
                )
            ]
        ),
    ]

    references = query_result_references_from_messages(messages)
    assert len(references) == 1
    assert references[0].query == "select 1"
    reference = find_query_result_reference(messages, "result_call.json")
    assert reference is not None
    resolved = await resolve_query_result(
        reference, store=store, context=QueryResultContext()
    )
    assert resolved.data == data
    assert resolved.source == "store"


@pytest.mark.asyncio
async def test_cli_gc_preserves_live_result_and_sweeps_old_orphan(tmp_path) -> None:
    now = 2_000_000.0
    store = FilesystemQueryResultStore(tmp_path / "results")
    data = b'{"success":true,"row_count":1,"results":[{"value":1}],"file":"result_call.json"}'
    live_draft = descriptor_for_data(
        data,
        result_id=new_query_result_id(),
        file="result_call.json",
        row_count=1,
        columns=("value",),
        created_at=now - 200_000,
    )
    orphan_draft = descriptor_for_data(
        data,
        result_id=new_query_result_id(),
        file="result_orphan.json",
        row_count=1,
        columns=("value",),
        created_at=now - 200_000,
    )
    live = await store.put(
        QueryResultData(data), descriptor=live_draft, context=QueryResultContext()
    )
    orphan = await store.put(
        QueryResultData(data), descriptor=orphan_draft, context=QueryResultContext()
    )
    messages = [
        ModelRequest(
            parts=[
                ToolReturnPart(
                    "execute_sql",
                    '{"success":true,"preview_rows":[]}',
                    "call",
                    metadata={"query_result": live.to_dict()},
                )
            ]
        )
    ]
    threads = ThreadStorage()
    threads.db_path = tmp_path / "threads.db"
    await threads.save_snapshot(
        messages_json=ModelMessagesTypeAdapter.dump_json(messages),
        database_name="test",
    )

    result = await collect_cli_query_results(
        threads, store, force=True, grace_seconds=86_400, now=now
    )

    assert result.complete is True
    assert result.deleted == 1
    assert (await store.get(live.id, context=QueryResultContext())).data == data
    with pytest.raises(QueryResultUnavailable):
        await store.get(orphan.id, context=QueryResultContext())


class _Database:
    display_name = "test"
    sqlglot_dialect = "sqlite"

    async def execute_query(self, *args, **kwargs):
        del args, kwargs
        return [{"value": index} for index in range(1000)]


@pytest.mark.asyncio
async def test_execute_sql_stores_complete_rows_but_returns_bounded_projection() -> (
    None
):
    store = InMemoryQueryResultStore()
    tool = ExecuteSQLTool(query_result_store=store)
    tool.db = _Database()  # type: ignore[assignment]
    tool.schema_manager = SimpleNamespace()  # type: ignore[assignment]
    ctx = SimpleNamespace(
        tool_call_id="call",
        run_id="run",
        conversation_id="conversation",
        metadata={"tenant_id": "acme"},
    )

    returned = await tool.execute(ctx, "select value from numbers")  # type: ignore[arg-type]

    assert isinstance(returned, ToolReturn)
    projection = json.loads(returned.return_value)
    assert len(returned.return_value.encode()) <= MAX_MODEL_QUERY_RESULT_BYTES
    assert projection["results_truncated"] is True
    descriptor = returned.metadata["query_result"]
    loaded = await store.get(descriptor["id"], context=QueryResultContext())
    assert len(loaded.rows()) == 1000
    assert loaded.descriptor.sha256 == hashlib.sha256(loaded.data).hexdigest()
