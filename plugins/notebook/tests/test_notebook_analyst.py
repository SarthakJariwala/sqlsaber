from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import nbformat
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from sqlsaber_notebook.analyst import (
    _cache_settings,
    analyze,
    supports_notebook_images,
)
from sqlsaber_notebook.execution.fake import FakeNotebookBackend, FakeRunResult
from sqlsaber_notebook.result import Workspace


def _executor(
    notebook: bytes,
    inputs: Mapping[str, bytes],
    run: int,
) -> FakeRunResult:
    del inputs
    return FakeRunResult(
        notebook,
        {"evidence.txt": f"run={run}".encode()},
    )


async def test_immediate_text_answer_is_allowed_and_usage_merges_once() -> None:
    backend = FakeNotebookBackend(_executor)
    parent_usage = RunUsage(requests=4)
    result = await analyze(
        "Give a preliminary answer",
        Workspace(()),
        model=TestModel(call_tools=[]),
        model_provider="test",
        backend=backend,
        parent_usage=parent_usage,
    )

    assert result.answer == "success (no tool calls)"
    assert nbformat.reads(result.notebook.decode(), as_version=4).cells == []
    assert result.files == []
    assert parent_usage.requests == 5
    assert backend.environments[0].closed is True


async def test_scripted_workspace_edit_then_final_answer() -> None:
    def respond(messages, info) -> ModelResponse:
        del info
        returns = [
            part
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        if not returns:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="list_workspace",
                        args={},
                        tool_call_id="workspace-call",
                    )
                ]
            )
        if returns[-1].tool_name == "list_workspace":
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="edit_cell",
                        args={"contents": "print('evidence')"},
                        tool_call_id="edit-call",
                    )
                ]
            )
        return ModelResponse(parts=[TextPart(content="The evidence was verified.")])

    backend = FakeNotebookBackend(_executor)
    result = await analyze(
        "Verify the evidence",
        Workspace(()),
        model=FunctionModel(respond),
        model_provider="test",
        backend=backend,
    )

    notebook = nbformat.reads(result.notebook.decode(), as_version=4)
    assert result.answer == "The evidence was verified."
    assert notebook.cells[0].source == "print('evidence')"
    assert result.files[0].name == "evidence.txt"
    assert result.files[0].data == b"run=1"
    assert result.provenance == ["cell:0"]
    assert backend.environments[0].closed is True


def test_model_settings_and_multimodal_allowlist() -> None:
    anthropic = cast(dict[str, Any], _cache_settings("anthropic"))
    generic = cast(dict[str, Any], _cache_settings("test"))
    assert anthropic is not None
    assert anthropic["parallel_tool_calls"] is False
    assert anthropic["anthropic_cache_instructions"] is True
    assert anthropic["anthropic_cache_tool_definitions"] is True
    assert "anthropic_cache" not in anthropic
    assert generic == {"parallel_tool_calls": False}
    assert supports_notebook_images("claude-sonnet-4", "anthropic") is True
    assert supports_notebook_images("unknown", "custom") is False
