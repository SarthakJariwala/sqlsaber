from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import nbformat
import pytest
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage, UsageLimits
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


async def test_direct_analysis_can_select_a_daytona_snapshot() -> None:
    backend = FakeNotebookBackend(_executor)
    backend.name = "daytona"

    result = await analyze(
        "Give an answer",
        Workspace(()),
        model=TestModel(call_tools=[]),
        model_provider="test",
        backend=backend,
        snapshot="analytics-ready",
    )

    assert result.answer == "success (no tool calls)"
    assert backend.snapshots == ["analytics-ready"]


async def test_analysis_uses_daytona_snapshot_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SQLSABER_NOTEBOOK_SNAPSHOT", "analytics-ready")
    backend = FakeNotebookBackend(_executor)
    backend.name = "daytona"

    await analyze(
        "Give an answer",
        Workspace(()),
        model=TestModel(call_tools=[]),
        model_provider="test",
        backend=backend,
    )

    assert backend.snapshots == ["analytics-ready"]


async def test_snapshot_rejects_non_daytona_backend() -> None:
    backend = FakeNotebookBackend(_executor)

    with pytest.raises(ValueError, match="only supported by Daytona"):
        await analyze(
            "Give an answer",
            Workspace(()),
            model=TestModel(call_tools=[]),
            model_provider="test",
            backend=backend,
            snapshot="analytics-ready",
        )

    assert backend.environments == []


async def test_image_and_snapshot_are_mutually_exclusive() -> None:
    backend = FakeNotebookBackend(_executor)
    backend.name = "daytona"

    with pytest.raises(ValueError, match="mutually exclusive"):
        await analyze(
            "Give an answer",
            Workspace(()),
            model=TestModel(call_tools=[]),
            model_provider="test",
            backend=backend,
            image="example/image:latest",
            snapshot="analytics-ready",
        )

    assert backend.environments == []


async def test_omitted_usage_limits_leave_direct_analysis_unlimited() -> None:
    backend = FakeNotebookBackend(_executor)
    parent_usage = RunUsage(requests=50)

    result = await analyze(
        "Give an answer",
        Workspace(()),
        model=TestModel(call_tools=[]),
        model_provider="test",
        backend=backend,
        parent_usage=parent_usage,
    )

    assert result.answer == "success (no tool calls)"
    assert parent_usage.requests == 51


async def test_exhausted_request_budget_skips_environment_provisioning() -> None:
    backend = FakeNotebookBackend(_executor)

    with pytest.raises(UsageLimitExceeded, match="request_limit of 5"):
        await analyze(
            "Inspect the workspace",
            Workspace(()),
            model=TestModel(call_tools=[]),
            model_provider="test",
            backend=backend,
            usage_limits=UsageLimits(request_limit=5),
            parent_usage=RunUsage(requests=5),
        )

    assert backend.environments == []


async def test_nested_analysis_shares_parent_budget_and_usage() -> None:
    def respond(messages, info) -> ModelResponse:
        del info
        if any(
            isinstance(part, ToolReturnPart)
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
        ):
            return ModelResponse(parts=[TextPart(content="finished")])
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="list_workspace",
                    args={},
                    tool_call_id="workspace-call",
                )
            ]
        )

    backend = FakeNotebookBackend(_executor)
    parent_usage = RunUsage(requests=4)
    usage_limits = UsageLimits(request_limit=5)

    with pytest.raises(UsageLimitExceeded, match="request_limit of 5"):
        await analyze(
            "Inspect the workspace",
            Workspace(()),
            model=FunctionModel(respond),
            model_provider="test",
            backend=backend,
            usage_limits=usage_limits,
            parent_usage=parent_usage,
        )

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
