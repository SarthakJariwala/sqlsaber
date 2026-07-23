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
from sqlsaber_notebook.result import AnalysisResult, ArtifactRef

from sqlsaber.artifacts import InMemoryArtifactPublisher
from sqlsaber.query_results import InMemoryQueryResultStore


def _ctx(messages: list[Any], *, tool_call_id: str = "analysis-call") -> Any:
    return SimpleNamespace(
        messages=messages,
        tool_call_id=tool_call_id,
        usage=RunUsage(),
        run_id="run-1",
        conversation_id="conversation-1",
        metadata={"tenant_id": "acme"},
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


class _RecordingTUI:
    def __init__(self) -> None:
        self.markdown: list[str] = []
        self.images: list[tuple[bytes, str, dict[str, object]]] = []
        self.panels = 0

    def append_panel(self) -> _RecordingTUI:
        self.panels += 1
        return self

    def append_markdown(self, text: str = "", *, muted: bool = False) -> object:
        del muted
        self.markdown.append(text)
        return object()

    def append_image(
        self,
        data: bytes,
        mime_type: str,
        *,
        filename: str | None = None,
        max_width_cells: int = 60,
        max_height_cells: int | None = None,
    ) -> object:
        self.images.append(
            (
                data,
                mime_type,
                {
                    "filename": filename,
                    "max_width_cells": max_width_cells,
                    "max_height_cells": max_height_cells,
                },
            )
        )
        return object()


@pytest.mark.asyncio
async def test_workspace_selects_newest_successful_selects_and_pairs_sql() -> None:
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

    workspace = await build_workspace_from_history(
        _ctx(messages),
        only=None,
        query_result_store=InMemoryQueryResultStore(),
    )

    assert [item.name for item in workspace.files] == [
        "result_new.json",
        "result_old.json",
    ]
    assert [item.sql for item in workspace.manifest] == [
        "select 2 as value",
        "select 1 as value",
    ]
    assert json.loads(workspace.files[0].data)["results"] == [{"value": 2}]


@pytest.mark.asyncio
async def test_workspace_explicit_selection_is_ordered_and_all_or_error() -> None:
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

    workspace = await build_workspace_from_history(
        _ctx(messages),
        only=["result_one.json", "result_two.json"],
        query_result_store=InMemoryQueryResultStore(),
    )
    assert [item.name for item in workspace.files] == [
        "result_one.json",
        "result_two.json",
    ]

    with pytest.raises(ValueError, match="not found: result_missing.json"):
        await build_workspace_from_history(
            _ctx(messages),
            only=["result_one.json", "result_missing.json"],
            query_result_store=InMemoryQueryResultStore(),
        )


@pytest.mark.asyncio
async def test_workspace_rejects_invalid_requested_keys() -> None:
    messages = _sql_exchange(
        "real",
        "select 1",
        {
            "success": True,
            "results": [{"value": 1}],
            "file": "result_different.json",
        },
    )
    workspace = await build_workspace_from_history(
        _ctx(messages),
        only=None,
        query_result_store=InMemoryQueryResultStore(),
    )
    assert [item.name for item in workspace.files] == ["result_different.json"]

    with pytest.raises(ValueError, match="Invalid SQL result file key"):
        await build_workspace_from_history(
            _ctx(messages),
            only=["../secret.json"],
            query_result_store=InMemoryQueryResultStore(),
        )


@pytest.mark.asyncio
async def test_analyze_tool_renders_notebook_and_child_answer(
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

    request_tui = _RecordingTUI()
    assert tool.render_executing_tui(request_tui, {"goal": "Calculate the total"})
    assert request_tui.panels == 1
    assert request_tui.markdown == ["**Analyzing data**\n\nCalculate the total"]

    tui = _RecordingTUI()
    assert tool.render_result_tui(
        tui,
        returned.return_value,
        tool_call_id="analysis-call",
        metadata=returned.metadata,
    )
    assert tui.panels == 1
    assert "Analysis notebook" in tui.markdown[0]
    assert "print('evidence')" in tui.markdown[0]
    assert tui.markdown[1] == "**Plot 1**"
    assert tui.images == [
        (
            _png_bytes(),
            "image/png",
            {
                "filename": "plot_1.png",
                "max_width_cells": 80,
                "max_height_cells": 24,
            },
        )
    ]
    assert "Analysis result" in tui.markdown[-1]
    assert "The calculated answer is 10" in tui.markdown[-1]
    assert not tool.render_result_tui(
        tui,
        returned.return_value,
        tool_call_id="analysis-call",
    )

    rich_returned = await tool.execute(
        _ctx(messages, tool_call_id="rich-analysis-call"), "Calculate the total"
    )
    assert isinstance(rich_returned, ToolReturn)
    console = Console(
        record=True, force_terminal=True, color_system="truecolor", width=60
    )
    assert tool.render_result_event(
        console,
        rich_returned.return_value,
        tool_call_id="rich-analysis-call",
        metadata=rich_returned.metadata,
    )
    rendered = console.export_text()
    assert "Analysis notebook" in rendered
    assert "print('evidence')" in rendered
    assert "evidence" in rendered
    assert "Plot 1" in rendered
    assert "Analysis result" in rendered
    assert "The calculated answer is 10" in rendered
    assert not tool.render_result_event(
        console,
        rich_returned.return_value,
        tool_call_id="rich-analysis-call",
    )


@pytest.mark.asyncio
async def test_analyze_tool_publishes_notebook_images_and_generated_files(
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
    publisher = InMemoryArtifactPublisher()
    context = SimpleNamespace(
        resolve_subagent_model=lambda *args, **kwargs: (
            "anthropic:claude-test",
            "anthropic:claude-test",
            "anthropic",
        ),
        artifact_publisher=publisher,
        artifact_failure_mode="required",
    )
    backend = SimpleNamespace(name="docker")
    captured: dict[str, Any] = {}

    async def fake_analyze(goal: str, workspace: Any, **kwargs: Any) -> AnalysisResult:
        captured.update(goal=goal, workspace=workspace, **kwargs)
        return AnalysisResult(
            answer="Published answer.",
            notebook=_notebook_bytes(),
            images=[_png_bytes()],
            files=[ArtifactRef("nested/evidence.txt", b"evidence", "text/plain")],
            provenance=["input:result_rows.json", "cell:0"],
        )

    monkeypatch.setattr(capability_module, "analyze", fake_analyze)
    monkeypatch.setattr(capability_module, "resolve_notebook_backend", lambda: backend)
    monkeypatch.setattr(
        capability_module, "resolve_notebook_image", lambda: "test-image"
    )

    returned = await AnalyzeDataTool(cast(Any, context)).execute(
        _ctx(messages), "Calculate the total"
    )

    assert isinstance(returned, ToolReturn)
    assert captured["collect_files"] is True
    assert returned.metadata["analysis_id"]
    assert [item["name"] for item in returned.metadata["artifacts"]] == [
        "analysis.ipynb",
        "plots/plot_1.png",
        "files/nested/evidence.txt",
    ]
    bundle, publication_context = publisher.publications[
        returned.metadata["analysis_id"]
    ]
    assert bundle.kind == "notebook-analysis"
    assert publication_context.run_id == "run-1"
    assert publication_context.conversation_id == "conversation-1"
    assert publication_context.metadata == {"tenant_id": "acme"}


@pytest.mark.parametrize("failure_mode", ["required", "best_effort"])
@pytest.mark.asyncio
async def test_analyze_tool_handles_artifact_publication_failure(
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
) -> None:
    class FailingPublisher:
        async def publish(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            raise RuntimeError("bucket unavailable")

    context = SimpleNamespace(
        resolve_subagent_model=lambda *args, **kwargs: (
            "anthropic:claude-test",
            "anthropic:claude-test",
            "anthropic",
        ),
        artifact_publisher=FailingPublisher(),
        artifact_failure_mode=failure_mode,
    )
    backend = SimpleNamespace(name="docker")

    async def fake_analyze(goal: str, workspace: Any, **kwargs: Any) -> AnalysisResult:
        del goal, workspace, kwargs
        return AnalysisResult(
            answer="Analysis answer.",
            notebook=_notebook_bytes(),
            images=[],
            files=[],
            provenance=["cell:0"],
        )

    monkeypatch.setattr(capability_module, "analyze", fake_analyze)
    monkeypatch.setattr(capability_module, "resolve_notebook_backend", lambda: backend)
    monkeypatch.setattr(
        capability_module, "resolve_notebook_image", lambda: "test-image"
    )
    messages = _sql_exchange(
        "rows",
        "select 1",
        {"success": True, "results": [{"value": 1}], "file": "result_rows.json"},
    )

    returned = await AnalyzeDataTool(cast(Any, context)).execute(
        _ctx(messages), "Analyze"
    )

    if failure_mode == "required":
        assert isinstance(returned, str)
        assert json.loads(returned)["phase"] == "artifact-publication"
    else:
        assert isinstance(returned, ToolReturn)
        assert returned.return_value == "Analysis answer."
        assert returned.metadata["artifact_error"] == "bucket unavailable"


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
