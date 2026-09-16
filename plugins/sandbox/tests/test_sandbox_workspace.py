"""Authorized SQL-history and attachment workspace construction tests."""

from __future__ import annotations

import json
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
)

from sqlsaber.query_results import InMemoryQueryResultStore
from sqlsaber.workspace_inputs import WorkspaceResolutionContext
from sqlsaber_sandbox.config import WorkspaceLimits
from sqlsaber_sandbox.result import WorkspaceFile
from sqlsaber_sandbox.workspace import build_workspace_from_history


def _ctx(messages: list[Any], *, conversation_id: str = "conversation-1") -> Any:
    return SimpleNamespace(
        messages=messages,
        run_id="run-1",
        conversation_id=conversation_id,
        tool_call_id="analysis-1",
        metadata={"tenant_id": "acme"},
    )


def _sql_exchange(tool_call_id: str, query: str, value: str) -> list[Any]:
    payload = {
        "success": True,
        "results": [{"value": value}],
        "file": f"result_{tool_call_id}.json",
    }
    return [
        ModelResponse(
            parts=[ToolCallPart("execute_sql", {"query": query}, tool_call_id)]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    "execute_sql",
                    json.dumps(payload),
                    tool_call_id,
                )
            ]
        ),
    ]


async def _build(
    messages: list[Any],
    *,
    only: list[str] | None,
    attachment_refs: list[str] | None = None,
    resolver: Any = None,
    limits: WorkspaceLimits = WorkspaceLimits(),
):
    return await build_workspace_from_history(
        _ctx(messages),
        only=only,
        attachment_refs=attachment_refs,
        query_result_store=InMemoryQueryResultStore(),
        workspace_input_resolver=resolver,
        limits=limits,
    )


@pytest.mark.asyncio
async def test_none_selects_only_the_newest_bounded_sql_results() -> None:
    messages = [
        *_sql_exchange("old", "select 'old'", "old"),
        *_sql_exchange("middle", "select 'middle'", "middle"),
        *_sql_exchange("new", "select 'new'", "new"),
    ]

    workspace = await _build(
        messages,
        only=None,
        limits=WorkspaceLimits(default_results=2),
    )

    assert [item.name for item in workspace.files] == [
        "result_new.json",
        "result_middle.json",
    ]
    assert [item.provenance["query"] for item in workspace.files] == [
        "select 'new'",
        "select 'middle'",
    ]


@pytest.mark.asyncio
async def test_empty_only_selects_no_sql_and_allows_an_empty_workspace() -> None:
    workspace = await _build(
        _sql_exchange("rows", "select 1", "one"),
        only=[],
        attachment_refs=[],
    )

    assert workspace.files == ()
    assert workspace.manifest_bytes() == b"[]"


@pytest.mark.asyncio
async def test_explicit_sql_and_opaque_attachments_share_one_workspace() -> None:
    captured: dict[str, object] = {}

    class Resolver:
        async def resolve(
            self,
            refs: Sequence[str],
            *,
            context: WorkspaceResolutionContext,
        ) -> list[WorkspaceFile]:
            captured.update(refs=refs, context=context)
            return [
                WorkspaceFile(
                    "chart.png",
                    b"png",
                    media_type="image/png",
                    provenance={"attachment_id": "attachment-1"},
                )
            ]

    workspace = await _build(
        _sql_exchange("rows", "select * from rows", "one"),
        only=["result_rows.json"],
        attachment_refs=["https://opaque.invalid/token"],
        resolver=Resolver(),
    )

    assert captured["refs"] == ["https://opaque.invalid/token"]
    context = captured["context"]
    assert isinstance(context, WorkspaceResolutionContext)
    assert context.conversation_id == "conversation-1"
    assert context.metadata == {"tenant_id": "acme"}
    assert [item.name for item in workspace.files] == [
        "result_rows.json",
        "chart.png",
    ]
    assert workspace.files[0].provenance == {"query": "select * from rows"}
    assert workspace.files[1].provenance == {"attachment_id": "attachment-1"}


@pytest.mark.asyncio
@pytest.mark.parametrize("refs", [[""], ["duplicate", "duplicate"], ["bad\nref"]])
async def test_invalid_attachment_refs_never_reach_the_resolver(
    refs: list[str],
) -> None:
    class Resolver:
        async def resolve(self, refs, *, context):
            del refs, context
            raise AssertionError("invalid refs must not reach the resolver")

    with pytest.raises(ValueError):
        await _build([], only=[], attachment_refs=refs, resolver=Resolver())


@pytest.mark.asyncio
async def test_cross_scope_resolver_failures_do_not_leak_adapter_details() -> None:
    class Resolver:
        async def resolve(self, refs, *, context):
            del refs, context
            raise PermissionError("tenant-b/private/customer-list.csv")

    with pytest.raises(
        ValueError, match="Attachment inputs could not be resolved"
    ) as exc:
        await _build(
            [],
            only=[],
            attachment_refs=["cross-scope-ref"],
            resolver=Resolver(),
        )

    assert "tenant-b" not in str(exc.value)
    assert exc.value.__cause__ is None


@pytest.mark.asyncio
@pytest.mark.parametrize("resolved", [b"bytes", [object()], []])
async def test_resolver_must_return_a_nonempty_sequence_of_protocol_files(
    resolved: object,
) -> None:
    class Resolver:
        async def resolve(self, refs, *, context):
            del refs, context
            return resolved

    with pytest.raises(ValueError, match="Attachment inputs could not be resolved"):
        await _build(
            [],
            only=[],
            attachment_refs=["opaque-ref"],
            resolver=Resolver(),
        )


@pytest.mark.asyncio
async def test_explicit_missing_and_oversized_sql_results_fail() -> None:
    messages = _sql_exchange("large", "select large", "x" * 100)

    with pytest.raises(ValueError, match="not found"):
        await _build(messages, only=["result_missing.json"])

    baseline = await _build(messages, only=["result_large.json"])
    size = len(baseline.files[0].data)
    with pytest.raises(ValueError, match=f"exceeds {size - 1} bytes"):
        await _build(
            messages,
            only=["result_large.json"],
            limits=WorkspaceLimits(max_file_bytes=size - 1),
        )


@pytest.mark.asyncio
async def test_automatic_sql_selection_stops_at_combined_count_and_byte_budgets() -> (
    None
):
    messages = _sql_exchange("rows", "select 1", "one")
    sql = await _build(messages, only=["result_rows.json"])
    sql_bytes = len(sql.files[0].data)

    class Resolver:
        async def resolve(self, refs, *, context):
            del refs, context
            return [WorkspaceFile("attachment.bin", b"abc")]

    byte_limited = await _build(
        messages,
        only=None,
        attachment_refs=["opaque-ref"],
        resolver=Resolver(),
        limits=WorkspaceLimits(max_total_bytes=sql_bytes + 2),
    )
    count_limited = await _build(
        messages,
        only=None,
        attachment_refs=["opaque-ref"],
        resolver=Resolver(),
        limits=WorkspaceLimits(max_files=1),
    )

    assert [item.name for item in byte_limited.files] == ["attachment.bin"]
    assert [item.name for item in count_limited.files] == ["attachment.bin"]


@pytest.mark.asyncio
async def test_sql_and_attachment_name_collisions_fail() -> None:
    class Resolver:
        async def resolve(self, refs, *, context):
            del refs, context
            return [WorkspaceFile("result_rows.json", b"collision")]

    with pytest.raises(ValueError, match="Duplicate workspace filename"):
        await _build(
            _sql_exchange("rows", "select 1", "one"),
            only=["result_rows.json"],
            attachment_refs=["opaque-ref"],
            resolver=Resolver(),
        )
