from dataclasses import replace

import pytest
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
)

from sqlsaber.query_result_resolution import (
    find_query_result_reference,
    query_result_references_from_messages,
    resolve_query_result,
)
from sqlsaber.query_results import (
    InMemoryQueryResultStore,
    QueryResultContext,
    QueryResultData,
    QueryResultUnavailable,
    descriptor_for_data,
)

FLUX_DATA = b'{"success":true,"file":"result_flux.json","row_count":2,"results":[{"value":"9007199254740993"},{"value":null}],"flux":{"query":"flux query"}}'


def exchange(tool, call, *, metadata=None, query="query", content=None):
    return [
        ModelResponse(parts=[ToolCallPart(tool, {"query": query}, call)]),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool,
                    content or {"success": True, "results": [{"value": "preview"}]},
                    call,
                    metadata=metadata,
                )
            ]
        ),
    ]


def descriptor(
    data=FLUX_DATA,
    *,
    result_id="qr_0123456789abcdef0123456789abcdef",
    file="result_flux.json",
):
    return descriptor_for_data(
        data, result_id=result_id, file=file, row_count=2, columns=("value",)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["execute_flux", "custom_query"])
async def test_descriptor_results_are_tool_independent_and_select_complete_bytes(
    tool_name,
):
    store = InMemoryQueryResultStore()
    retained = await store.put(
        QueryResultData(FLUX_DATA),
        descriptor=descriptor(),
        context=QueryResultContext(),
    )
    messages = [
        *exchange(
            "execute_sql",
            "sql",
            content={
                "success": True,
                "file": "result_sql.json",
                "results": [{"value": 1}],
            },
        ),
        *exchange(
            tool_name,
            "flux",
            metadata={"query_result": retained.to_dict()},
            query="flux query",
            content={"success": True, "rows": [], "rows_omitted": True},
        ),
        *exchange("other_tool", "ignore"),
        *exchange(
            tool_name, "duplicate", metadata={"query_result": retained.to_dict()}
        ),
    ]
    references = query_result_references_from_messages(messages)
    assert [(ref.file, ref.tool_name) for ref in references] == [
        ("result_sql.json", "execute_sql"),
        ("result_flux.json", tool_name),
    ]
    for selector in ("result_flux.json", "flux", retained.id):
        ref = find_query_result_reference(messages, selector)
        assert ref == references[1]
        resolved = await resolve_query_result(
            ref, store=store, context=QueryResultContext()
        )
        assert resolved.data == FLUX_DATA
        assert resolved.source == "store"
    assert references[1].query == "flux query"
    assert (
        await resolve_query_result(
            references[0], store=store, context=QueryResultContext()
        )
    ).payload()["results"] == [{"value": 1}]


def test_query_pairing_requires_both_tool_name_and_call_id():
    messages = [
        *exchange(
            "execute_flux",
            "same",
            metadata={"query_result": descriptor().to_dict()},
            query="flux query",
        ),
        ModelResponse(
            parts=[ToolCallPart("execute_sql", {"query": "sql query"}, "same")]
        ),
    ]
    assert query_result_references_from_messages(messages)[0].query == "flux query"
    messages[0] = ModelResponse(
        parts=[ToolCallPart("different_tool", {"query": "wrong query"}, "same")]
    )
    assert query_result_references_from_messages(messages)[0].query is None


@pytest.mark.parametrize("tool", ["execute_sql", "execute_flux", "other_tool"])
def test_malformed_descriptor_never_falls_back_to_inline_results(tool):
    messages = [
        *exchange(tool, "bad", metadata={"query_result": {"id": "bad"}}),
        *exchange("execute_sql", "good"),
    ]
    assert [
        ref.tool_call_id for ref in query_result_references_from_messages(messages)
    ] == ["good"]


def test_duplicate_file_selector_is_ambiguous():
    messages = [
        *exchange(
            "execute_flux", "flux", metadata={"query_result": descriptor().to_dict()}
        ),
        *exchange(
            "execute_sql",
            "sql",
            metadata={
                "query_result": descriptor(
                    result_id="qr_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                ).to_dict()
            },
        ),
    ]
    with pytest.raises(QueryResultUnavailable):
        find_query_result_reference(messages, "result_flux.json")
    assert find_query_result_reference(messages, "flux").tool_name == "execute_flux"


@pytest.mark.asyncio
async def test_descriptor_unavailable_or_mismatched_does_not_use_preview():
    retained = descriptor()
    messages = exchange(
        "execute_flux", "flux", metadata={"query_result": retained.to_dict()}
    )
    reference = query_result_references_from_messages(messages)[0]
    store = InMemoryQueryResultStore()
    with pytest.raises(QueryResultUnavailable):
        await resolve_query_result(reference, store=store, context=QueryResultContext())
    await store.put(
        QueryResultData(FLUX_DATA), descriptor=retained, context=QueryResultContext()
    )
    with pytest.raises(QueryResultUnavailable):
        await resolve_query_result(
            replace(
                reference, descriptor=replace(retained, file="result_swapped.json")
            ),
            store=store,
            context=QueryResultContext(),
        )
    assert (
        await resolve_query_result(reference, store=store, context=QueryResultContext())
    ).data == FLUX_DATA
