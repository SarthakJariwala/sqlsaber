"""Managed SQLsaber notebook capability tests."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, cast

import nbformat
import pytest
from PIL import Image
from pydantic_ai import ToolReturn
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.usage import RunUsage
from rich.console import Console
from sqlsaber_notebook import capability as capability_module
from sqlsaber_notebook.capability import (
    AnalyzeDataTool,
    Notebook,
    build_workspace_from_history,
)
from sqlsaber_notebook.execution import NotebookBackendUnavailable
from sqlsaber_notebook.result import AnalysisResult


def _ctx(messages: list[Any], *, tool_call_id: str = "analysis-call") -> Any:
    return SimpleNamespace(
        messages=messages,
        tool_call_id=tool_call_id,
        usage=RunUsage(),
    )


def _sql_exchange(
    tool_call_id: str,
    query: str,
    payload: dict[str, Any],
) -> list[Any]:
    return [
        ModelResponse(
            parts=[
                ToolCallPart(
                    "execute_sql",
                    {"query": query},
                    tool_call_id=tool_call_id,
                )
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    "execute_sql",
                    json.dumps(payload),
                    tool_call_id=tool_call_id,
                )
            ]
        ),
    ]


def _notebook_bytes() -> bytes:
    notebook = nbformat.v4.new_notebook()
    notebook.cells = [
        nbformat.v4.new_code_cell(
            "print('evidence')",
            outputs=[
                nbformat.v4.new_output("stream", name="stdout", text="evidence\n")
            ],
        )
    ]
    return nbformat.writes(notebook).encode()


def _png_bytes() -> bytes:
    import io

    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), "red").save(buffer, format="PNG")
    return buffer.getvalue()


def test_workspace_selects_newest_successful_selects_and_pairs_sql() -> None:
    messages = [
        *_sql_exchange(
            "old",
            "select 1 as value",
            {
                "success": True,
                "results": [{"value": 1}],
                "file": "result_old.json",
            },
        ),
        *_sql_exchange(
            "dml",
            "delete from values",
            {"success": True, "file": "result_dml.json"},
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    "other_tool",
                    {"success": True, "results": ["ignore"]},
                    tool_call_id="other",
                )
            ]
        ),
        *_sql_exchange(
            "new",
            "select 2 as value",
            {
                "success": True,
                "results": [{"value": 2}],
                "file": "result_new.json",
            },
        ),
    ]

    workspace = build_workspace_from_history(_ctx(messages), only=None)

    assert [item.name for item in workspace.files] == [
        "result_new.json",
        "result_old.json",
    ]
    assert [item.sql for item in workspace.manifest] == [
        "select 2 as value",
        "select 1 as value",
    ]
    assert json.loads(workspace.files[0].data)["results"] == [{"value": 2}]


def test_workspace_explicit_selection_is_ordered_and_all_or_error() -> None:
    messages = [
        *_sql_exchange(
            "one",
            "select 1",
            {"success": True, "results": [], "file": "result_one.json"},
        ),
        *_sql_exchange(
            "two",
            "select 2",
            {"success": True, "results": [], "file": "result_two.json"},
        ),
    ]

    workspace = build_workspace_from_history(
        _ctx(messages), only=["result_one.json", "result_two.json"]
    )
    assert [item.name for item in workspace.files] == [
        "result_one.json",
        "result_two.json",
    ]

    with pytest.raises(ValueError, match="not found: result_missing.json"):
        build_workspace_from_history(
            _ctx(messages), only=["result_one.json", "result_missing.json"]
        )


def test_workspace_rejects_forged_or_invalid_requested_keys() -> None:
    messages = _sql_exchange(
        "real",
        "select 1",
        {
            "success": True,
            "results": [{"value": 1}],
            "file": "result_different.json",
        },
    )
    with pytest.raises(ValueError, match="No successful row-returning"):
        build_workspace_from_history(_ctx(messages), only=None)

    with pytest.raises(ValueError, match="Invalid SQL result file key"):
        build_workspace_from_history(_ctx(messages), only=["../secret.json"])


@pytest.mark.asyncio
async def test_analyze_tool_returns_text_only_and_renders_ephemeral_notebook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages = _sql_exchange(
        "rows",
        "select * from sales",
        {
            "success": True,
            "results": [{"amount": 10}],
            "file": "result_rows.json",
        },
    )
    context = SimpleNamespace(
        resolve_subagent_model=lambda *args, **kwargs: (
            "anthropic:claude-test",
            "anthropic:claude-test",
            "anthropic",
        )
    )
    backend = SimpleNamespace(name="docker")
    captured: dict[str, Any] = {}

    async def fake_analyze(goal: str, workspace: Any, **kwargs: Any) -> AnalysisResult:
        captured.update(goal=goal, workspace=workspace, **kwargs)
        return AnalysisResult(
            answer="The calculated answer is 10.",
            notebook=_notebook_bytes(),
            images=[_png_bytes()],
            files=[],
            provenance=["input:result_rows.json", "cell:0"],
        )

    monkeypatch.setattr(capability_module, "analyze", fake_analyze)
    monkeypatch.setattr(capability_module, "resolve_notebook_backend", lambda: backend)
    monkeypatch.setattr(
        capability_module, "resolve_notebook_image", lambda: "test-image"
    )
    tool = AnalyzeDataTool(cast(Any, context))

    returned = await tool.execute(_ctx(messages), "Calculate the total")

    assert isinstance(returned, ToolReturn)
    assert returned.return_value == "The calculated answer is 10."
    assert returned.content is None
    assert returned.metadata["files"] == ["result_rows.json"]
    assert captured["collect_files"] is False
    assert captured["parent_usage"] is not None

    console = Console(
        record=True, force_terminal=True, color_system="truecolor", width=60
    )
    assert tool.render_result_event(
        console,
        returned.return_value,
        tool_call_id="analysis-call",
        metadata=returned.metadata,
    )
    rendered = console.export_text()
    assert "Analysis notebook" in rendered
    assert "print('evidence')" in rendered
    assert "evidence" in rendered
    assert "Plot 1" in rendered
    assert "The calculated answer is 10" not in rendered
    assert not tool.render_result_event(
        console,
        returned.return_value,
        tool_call_id="analysis-call",
    )


@pytest.mark.asyncio
async def test_analyze_tool_maps_backend_failure_to_bounded_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = SimpleNamespace(
        resolve_subagent_model=lambda *args, **kwargs: (
            "anthropic:claude-test",
            "anthropic:claude-test",
            "anthropic",
        )
    )
    monkeypatch.setattr(
        capability_module,
        "resolve_notebook_backend",
        lambda: (_ for _ in ()).throw(
            NotebookBackendUnavailable(
                "Docker is unavailable",
                backend="docker",
                phase="availability",
            )
        ),
    )
    tool = AnalyzeDataTool(cast(Any, context))

    messages = _sql_exchange(
        "rows",
        "select 1",
        {"success": True, "results": [{"value": 1}], "file": "result_rows.json"},
    )
    returned = await tool.execute(_ctx(messages), "Analyze")

    assert isinstance(returned, str)
    assert json.loads(returned) == {
        "error": "Docker is unavailable",
        "backend": "docker",
        "phase": "availability",
    }


def test_installed_capability_is_always_registered() -> None:
    notebook = capability_module.capability(cast(Any, SimpleNamespace()))
    assert isinstance(notebook, Notebook)
    assert notebook.tool.name == "analyze_data"
