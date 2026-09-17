import json
from types import SimpleNamespace

import pytest
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
)

from sqlsaber.query_results import (
    InMemoryQueryResultStore,
    QueryResultContext,
    QueryResultData,
    descriptor_for_data,
)
from sqlsaber_notebook.capability import build_workspace_from_history
from sqlsaber_notebook.config import WorkspaceLimits
from sqlsaber_notebook.execution import NotebookLimitExceeded
from sqlsaber_notebook.result import workspace_manifest_bytes


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "only, expected",
    [
        (None, ["result_flux.json", "result_sql.json"]),
        (["result_flux.json"], ["result_flux.json"]),
        (
            ["result_sql.json", "result_flux.json"],
            ["result_sql.json", "result_flux.json"],
        ),
    ],
)
async def test_mixed_history_resolves_full_flux_bytes_and_preserves_source(
    only, expected
):
    data = b'{"success":true,"file":"result_flux.json","row_count":2,"results":[{"value":"9007199254740993"},{"value":null}],"flux":{"query":"flux query","schemas":[],"tables":[]}}'
    store = InMemoryQueryResultStore()
    descriptor = await store.put(
        QueryResultData(data),
        descriptor=descriptor_for_data(
            data,
            result_id="qr_0123456789abcdef0123456789abcdef",
            file="result_flux.json",
            row_count=2,
            columns=("value",),
        ),
        context=QueryResultContext(),
    )
    messages = [
        ModelResponse(
            parts=[ToolCallPart("execute_sql", {"query": "select 1"}, "sql")]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    "execute_sql",
                    {
                        "success": True,
                        "file": "result_sql.json",
                        "results": [{"value": 1}],
                    },
                    "sql",
                )
            ]
        ),
        ModelResponse(
            parts=[ToolCallPart("execute_flux", {"query": "flux query"}, "flux")]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    "execute_flux",
                    {"success": True, "rows": [], "rows_omitted": True},
                    "flux",
                    metadata={"query_result": descriptor.to_dict()},
                )
            ]
        ),
    ]
    ctx = SimpleNamespace(messages=messages, metadata={"owner_id": 42})
    workspace = await build_workspace_from_history(ctx, only, query_result_store=store)
    assert [file.name for file in workspace.files] == expected
    flux_file = next(
        file for file in workspace.files if file.name == "result_flux.json"
    )
    assert flux_file.data == data
    manifest = json.loads(workspace_manifest_bytes(workspace))
    flux_entry = next(
        entry for entry in manifest if entry["file"] == "../inputs/result_flux.json"
    )
    assert flux_entry["sql"] is None
    assert flux_entry["source"] == "execute_flux"
    sql_entry = next(
        (entry for entry in manifest if entry["file"] == "../inputs/result_sql.json"),
        None,
    )
    if sql_entry:
        assert sql_entry["sql"] == "select 1"
    with pytest.raises(NotebookLimitExceeded, match="exceeds"):
        await build_workspace_from_history(
            ctx,
            ["result_flux.json"],
            query_result_store=store,
            limits=WorkspaceLimits(max_file_bytes=len(data) - 1),
        )
