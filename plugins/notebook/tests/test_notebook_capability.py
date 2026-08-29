"""Managed SQLsaber notebook capability tests."""

from __future__ import annotations

import json
from collections.abc import Sequence
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
from pydantic_ai.usage import RunUsage, UsageLimits
from sqlsaber_notebook import capability as capability_module
from sqlsaber_notebook.capability import (
    AnalyzeDataTool,
    Notebook,
    WorkspaceInputResolver,
    WorkspaceInputUnavailable,
    WorkspaceResolutionContext,
    build_workspace_from_history,
)
from sqlsaber_notebook.execution import (
    NotebookBackendUnavailable,
    NotebookExecutionError,
)
from sqlsaber.artifacts import InMemoryArtifactStore
from sqlsaber.query_results import InMemoryQueryResultStore
from sqlsaber.render.blocks import Image as ImageBlock
from sqlsaber.render.blocks import Md, Panel
from sqlsaber.run_usage import bind_usage_limits
from sqlsaber.tools.renderer import ToolRenderContext
from sqlsaber_notebook.result import AnalysisResult, ArtifactRef, WorkspaceFile


def _ctx(messages: list[Any], *, tool_call_id: str = "analysis-call") -> Any:
    return SimpleNamespace(
        messages=messages,
        tool_call_id=tool_call_id,
        usage=RunUsage(),
        usage_limits=UsageLimits(request_limit=200),
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
        max_width_cells: int | None = 60,
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
async def test_workspace_supports_attachment_only_analysis_with_resolver_context() -> (
    None
):
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
                    "preview.jpeg",
                    b"jpeg",
                    media_type="image/jpeg",
                    provenance={"attachment_id": "attachment-1"},
                )
            ]

    resolver: WorkspaceInputResolver = Resolver()
    workspace = await build_workspace_from_history(
        _ctx([]),
        only=None,
        attachment_refs=["opaque-ref-1"],
        workspace_input_resolver=resolver,
        query_result_store=InMemoryQueryResultStore(),
    )

    assert captured["refs"] == ["opaque-ref-1"]
    resolution_context = captured["context"]
    assert isinstance(resolution_context, WorkspaceResolutionContext)
    assert resolution_context.run_id == "run-1"
    assert resolution_context.conversation_id == "conversation-1"
    assert resolution_context.tool_call_id == "analysis-call"
    assert resolution_context.metadata == {"tenant_id": "acme"}
    assert [item.name for item in workspace.files] == ["preview.jpeg"]
    assert workspace.manifest[0].media_type == "image/jpeg"
    assert workspace.manifest[0].provenance == {"attachment_id": "attachment-1"}


@pytest.mark.asyncio
async def test_workspace_merges_sql_then_resolved_inputs_in_stable_order() -> None:
    class Resolver:
        async def resolve(self, refs, *, context):
            del refs, context
            return [
                WorkspaceFile("second.csv", b"second"),
                WorkspaceFile("first.csv", b"first"),
            ]

    messages = _sql_exchange(
        "rows",
        "select * from sales",
        {
            "success": True,
            "results": [{"amount": 10}],
            "file": "result_rows.json",
        },
    )
    workspace = await build_workspace_from_history(
        _ctx(messages),
        only=["result_rows.json"],
        attachment_refs=["opaque-b", "opaque-a"],
        workspace_input_resolver=Resolver(),
        query_result_store=InMemoryQueryResultStore(),
    )

    assert [item.name for item in workspace.files] == [
        "result_rows.json",
        "second.csv",
        "first.csv",
    ]
    assert [item.file for item in workspace.manifest] == [
        "result_rows.json",
        "second.csv",
        "first.csv",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("resolved", "error"),
    [
        (
            [
                WorkspaceFile("duplicate.csv", b"one"),
                WorkspaceFile("duplicate.csv", b"two"),
            ],
            "Duplicate workspace filename: duplicate.csv",
        ),
        ([WorkspaceFile("manifest.json", b"reserved")], "reserved"),
        ([WorkspaceFile("../unsafe.csv", b"unsafe")], "Unsafe workspace filename"),
    ],
)
async def test_workspace_rejects_invalid_resolved_filenames(
    resolved: list[WorkspaceFile],
    error: str,
) -> None:
    class Resolver:
        async def resolve(self, refs, *, context):
            del refs, context
            return resolved

    with pytest.raises(NotebookExecutionError, match=error):
        await build_workspace_from_history(
            _ctx([]),
            only=None,
            attachment_refs=["opaque-ref"],
            workspace_input_resolver=Resolver(),
            query_result_store=InMemoryQueryResultStore(),
        )


@pytest.mark.asyncio
async def test_workspace_rejects_resolved_filename_collision_with_sql() -> None:
    class Resolver:
        async def resolve(self, refs, *, context):
            del refs, context
            return [WorkspaceFile("result_rows.json", b"collision")]

    messages = _sql_exchange(
        "rows",
        "select 1",
        {"success": True, "results": [{"value": 1}], "file": "result_rows.json"},
    )
    with pytest.raises(
        NotebookExecutionError,
        match="Duplicate workspace filename: result_rows.json",
    ):
        await build_workspace_from_history(
            _ctx(messages),
            only=["result_rows.json"],
            attachment_refs=["opaque-ref"],
            workspace_input_resolver=Resolver(),
            query_result_store=InMemoryQueryResultStore(),
        )


@pytest.mark.asyncio
async def test_workspace_requires_a_resolver_for_attachment_refs() -> None:
    with pytest.raises(ValueError, match="No workspace input resolver is configured"):
        await build_workspace_from_history(
            _ctx([]),
            only=None,
            attachment_refs=["opaque-ref"],
            query_result_store=InMemoryQueryResultStore(),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("refs", [[], [""], ["duplicate", "duplicate"], ["bad\nref"]])
async def test_workspace_rejects_invalid_attachment_refs(refs: list[str]) -> None:
    class Resolver:
        async def resolve(self, refs, *, context):
            del refs, context
            raise AssertionError("invalid references must not reach the resolver")

    with pytest.raises(ValueError):
        await build_workspace_from_history(
            _ctx([]),
            only=None,
            attachment_refs=refs,
            workspace_input_resolver=Resolver(),
            query_result_store=InMemoryQueryResultStore(),
        )


@pytest.mark.asyncio
async def test_workspace_preserves_safe_unknown_ref_errors_and_hides_failures() -> None:
    class UnknownResolver:
        async def resolve(self, refs, *, context):
            del refs, context
            raise WorkspaceInputUnavailable(
                "Unknown or unauthorized attachment reference"
            )

    with pytest.raises(
        WorkspaceInputUnavailable,
        match="Unknown or unauthorized attachment reference",
    ):
        await build_workspace_from_history(
            _ctx([]),
            only=None,
            attachment_refs=["invented-ref"],
            workspace_input_resolver=UnknownResolver(),
            query_result_store=InMemoryQueryResultStore(),
        )

    class FailingResolver:
        async def resolve(self, refs, *, context):
            del refs, context
            raise RuntimeError("secret bucket provider failure")

    with pytest.raises(
        WorkspaceInputUnavailable,
        match="Attachment inputs could not be resolved",
    ) as captured:
        await build_workspace_from_history(
            _ctx([]),
            only=None,
            attachment_refs=["opaque-ref"],
            workspace_input_resolver=FailingResolver(),
            query_result_store=InMemoryQueryResultStore(),
        )
    assert "secret bucket" not in str(captured.value)

    class FailingSequence(Sequence[WorkspaceFile]):
        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> WorkspaceFile:
            del index
            raise RuntimeError("secret lazy storage failure")

    class LazyFailingResolver:
        async def resolve(self, refs, *, context):
            del refs, context
            return FailingSequence()

    with pytest.raises(
        WorkspaceInputUnavailable,
        match="Attachment inputs could not be resolved",
    ) as lazy_captured:
        await build_workspace_from_history(
            _ctx([]),
            only=None,
            attachment_refs=["opaque-ref"],
            workspace_input_resolver=LazyFailingResolver(),
            query_result_store=InMemoryQueryResultStore(),
        )
    assert "secret lazy" not in str(lazy_captured.value)


@pytest.mark.asyncio
async def test_workspace_rejects_invalid_resolver_output() -> None:
    class Resolver:
        async def resolve(self, refs, *, context):
            del refs, context
            return [object()]

    with pytest.raises(WorkspaceInputUnavailable, match="invalid file"):
        await build_workspace_from_history(
            _ctx([]),
            only=None,
            attachment_refs=["opaque-ref"],
            workspace_input_resolver=Resolver(),
            query_result_store=InMemoryQueryResultStore(),
        )


@pytest.mark.asyncio
async def test_workspace_enforces_file_count_across_sql_and_resolved_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(capability_module, "MAX_WORKSPACE_FILES", 1)

    class Resolver:
        async def resolve(self, refs, *, context):
            del refs, context
            return [WorkspaceFile("attachment.csv", b"attachment")]

    messages = _sql_exchange(
        "rows",
        "select 1",
        {"success": True, "results": [{"value": 1}], "file": "result_rows.json"},
    )
    with pytest.raises(NotebookExecutionError, match="more than 1 files"):
        await build_workspace_from_history(
            _ctx(messages),
            only=["result_rows.json"],
            attachment_refs=["opaque-ref"],
            workspace_input_resolver=Resolver(),
            query_result_store=InMemoryQueryResultStore(),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("include_sql", [False, True])
async def test_workspace_enforces_per_file_limit_for_each_input_class(
    monkeypatch: pytest.MonkeyPatch,
    include_sql: bool,
) -> None:
    monkeypatch.setattr(capability_module, "MAX_WORKSPACE_FILE_BYTES", 3)
    if include_sql:
        messages = _sql_exchange(
            "rows",
            "select 1",
            {
                "success": True,
                "results": [{"value": 1}],
                "file": "result_rows.json",
            },
        )
        kwargs = {"only": ["result_rows.json"]}
    else:
        messages = []
        kwargs = {"only": None}

    class Resolver:
        async def resolve(self, refs, *, context):
            del refs, context
            return [WorkspaceFile("attachment.bin", b"four")]

    with pytest.raises(NotebookExecutionError, match="exceeds 3 bytes"):
        await build_workspace_from_history(
            _ctx(messages),
            attachment_refs=None if include_sql else ["opaque-ref"],
            workspace_input_resolver=Resolver(),
            query_result_store=InMemoryQueryResultStore(),
            **kwargs,
        )


@pytest.mark.asyncio
async def test_workspace_enforces_aggregate_bytes_across_sql_and_resolved_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Resolver:
        async def resolve(self, refs, *, context):
            del refs, context
            return [WorkspaceFile("attachment.bin", b"attachment")]

    messages = _sql_exchange(
        "rows",
        "select 1",
        {"success": True, "results": [{"value": 1}], "file": "result_rows.json"},
    )
    baseline = await build_workspace_from_history(
        _ctx(messages),
        only=["result_rows.json"],
        attachment_refs=["opaque-ref"],
        workspace_input_resolver=Resolver(),
        query_result_store=InMemoryQueryResultStore(),
    )
    combined_bytes = sum(len(item.data) for item in baseline.files)
    monkeypatch.setattr(
        capability_module,
        "MAX_WORKSPACE_TOTAL_BYTES",
        combined_bytes - 1,
    )

    with pytest.raises(NotebookExecutionError, match="total bytes"):
        await build_workspace_from_history(
            _ctx(messages),
            only=["result_rows.json"],
            attachment_refs=["opaque-ref"],
            workspace_input_resolver=Resolver(),
            query_result_store=InMemoryQueryResultStore(),
        )


@pytest.mark.asyncio
async def test_workspace_enforces_reserved_manifest_byte_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(capability_module, "MAX_WORKSPACE_MANIFEST_BYTES", 100)

    class Resolver:
        async def resolve(self, refs, *, context):
            del refs, context
            return [
                WorkspaceFile(
                    "attachment.bin",
                    b"data",
                    provenance={"description": "x" * 200},
                )
            ]

    with pytest.raises(NotebookExecutionError, match="manifest exceeds 100 bytes"):
        await build_workspace_from_history(
            _ctx([]),
            only=None,
            attachment_refs=["opaque-ref"],
            workspace_input_resolver=Resolver(),
            query_result_store=InMemoryQueryResultStore(),
        )


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
async def test_analyze_tool_runs_attachment_only_analysis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Resolver:
        async def resolve(self, refs, *, context):
            del context
            assert refs == ["opaque-input"]
            return [WorkspaceFile("measurements.npz", b"npz")]

    context = SimpleNamespace(
        workspace_input_resolver=Resolver(),
        resolve_subagent_model=lambda *args, **kwargs: (
            "anthropic:claude-test",
            "anthropic:claude-test",
            "anthropic",
        ),
    )
    backend = SimpleNamespace(name="docker")
    captured: dict[str, Any] = {}

    async def fake_analyze(goal: str, workspace: Any, **kwargs: Any) -> AnalysisResult:
        captured.update(goal=goal, workspace=workspace, **kwargs)
        return AnalysisResult(
            answer="Attachment analyzed.",
            notebook=_notebook_bytes(),
            images=[],
            files=[],
            provenance=["input:measurements.npz"],
        )

    monkeypatch.setattr(capability_module, "analyze", fake_analyze)
    monkeypatch.setattr(capability_module, "resolve_notebook_backend", lambda: backend)
    monkeypatch.setattr(
        capability_module, "resolve_notebook_image", lambda: "test-image"
    )

    returned = await AnalyzeDataTool(cast(Any, context)).execute_with_attachments(
        _ctx([]),
        "Analyze the measurements",
        attachment_refs=["opaque-input"],
    )

    assert isinstance(returned, ToolReturn)
    assert returned.return_value == "Attachment analyzed."
    assert returned.metadata["files"] == ["measurements.npz"]
    assert [item.name for item in captured["workspace"].files] == ["measurements.npz"]


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
    run_ctx = _ctx(messages)

    returned = await tool.execute(run_ctx, "Calculate the total")

    assert isinstance(returned, ToolReturn)
    assert returned.return_value == "The calculated answer is 10."
    assert returned.content is None
    assert returned.metadata["files"] == ["result_rows.json"]
    assert captured["collect_files"] is False
    assert captured["usage_limits"].request_limit is None
    assert captured["parent_usage"] is run_ctx.usage

    executing = tool.render_executing({"goal": "Calculate the total"})
    assert executing is not None
    assert isinstance(executing[0], Panel)
    assert any(
        isinstance(child, Md) and "Calculate the total" in child.text
        for child in executing[0].blocks
    )

    result_blocks = tool.render_result(
        returned.return_value,
        context=ToolRenderContext(
            tool_call_id="analysis-call",
            metadata=returned.metadata,
        ),
    )
    assert result_blocks is not None
    panel = result_blocks[0]
    assert isinstance(panel, Panel)
    texts = [child.text for child in panel.blocks if isinstance(child, Md)]
    assert any("Analysis notebook" in text for text in texts)
    assert any("print('evidence')" in text for text in texts)
    assert any(text == "**Plot 1**" for text in texts)
    images = [child for child in panel.blocks if isinstance(child, ImageBlock)]
    assert images[0].data == _png_bytes()
    assert images[0].filename == "plot_1.png"
    assert any("Analysis result" in text for text in texts)
    assert any("The calculated answer is 10" in text for text in texts)
    assert (
        tool.render_result(
            returned.return_value,
            context=ToolRenderContext(tool_call_id="analysis-call"),
        )
        is None
    )

    rich_returned = await tool.execute(
        _ctx(messages, tool_call_id="rich-analysis-call"), "Calculate the total"
    )
    assert isinstance(rich_returned, ToolReturn)
    replay = tool.render_result(
        rich_returned.return_value,
        context=ToolRenderContext(
            tool_call_id="rich-analysis-call",
            metadata=rich_returned.metadata,
        ),
    )
    assert replay is not None
    replay_text = "\n".join(
        child.text for child in replay[0].blocks if isinstance(child, Md)
    )
    assert "Analysis notebook" in replay_text
    assert "print('evidence')" in replay_text
    assert "Plot 1" in replay_text
    assert "Analysis result" in replay_text
    assert "The calculated answer is 10" in replay_text
    assert (
        tool.render_result(
            rich_returned.return_value,
            context=ToolRenderContext(tool_call_id="rich-analysis-call"),
        )
        is None
    )


def test_nested_usage_limits_are_unlimited_without_explicit_parent_limits() -> None:
    nested = capability_module._nested_usage_limits()

    assert nested == UsageLimits(request_limit=None)


def test_nested_usage_limits_inherit_explicit_parent_budget() -> None:
    parent = UsageLimits(
        request_limit=200,
        tool_calls_limit=10,
        total_tokens_limit=1_000_000,
    )

    with bind_usage_limits(parent):
        nested = capability_module._nested_usage_limits()

    assert nested.request_limit == 200
    assert nested.tool_calls_limit == 9
    assert nested.total_tokens_limit == 1_000_000
    assert parent.tool_calls_limit == 10


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

    class RecordingStore(InMemoryArtifactStore):
        published: tuple[Any, Any] | None = None

        async def publish(self, bundle: Any, *, context: Any):
            self.published = (bundle, context)
            return await super().publish(bundle, context=context)

    store = RecordingStore()
    context = SimpleNamespace(
        resolve_subagent_model=lambda *args, **kwargs: (
            "anthropic:claude-test",
            "anthropic:claude-test",
            "anthropic",
        ),
        artifact_store=store,
        artifact_failure_mode="required",
    )
    backend = SimpleNamespace(name="docker")
    captured: dict[str, Any] = {}
    notebook_bytes = _notebook_bytes()
    png_bytes = _png_bytes()

    async def fake_analyze(goal: str, workspace: Any, **kwargs: Any) -> AnalysisResult:
        captured.update(goal=goal, workspace=workspace, **kwargs)
        return AnalysisResult(
            answer="Published answer.",
            notebook=notebook_bytes,
            images=[png_bytes],
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
    reference = returned.metadata["artifact_publication"]
    assert reference["id"]
    assert reference["kind"] == "notebook-analysis"
    assert [item["name"] for item in reference["artifacts"]] == [
        "analysis.ipynb",
        "plots/plot_1.png",
        "files/nested/evidence.txt",
    ]
    assert store.published is not None
    bundle, publication_context = store.published
    assert bundle.kind == "notebook-analysis"
    assert bundle.metadata == {"provenance": ["input:result_rows.json", "cell:0"]}
    loaded = [
        await store.get(artifact["id"], context=publication_context)
        for artifact in reference["artifacts"]
    ]
    assert [artifact.data for artifact in loaded] == [
        notebook_bytes,
        png_bytes,
        b"evidence",
    ]
    assert publication_context.run_id == "run-1"
    assert publication_context.conversation_id == "conversation-1"
    assert publication_context.metadata == {"tenant_id": "acme"}


@pytest.mark.parametrize("failure_mode", ["required", "best_effort"])
@pytest.mark.asyncio
async def test_analyze_tool_handles_artifact_publication_failure(
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
) -> None:
    class FailingStore:
        async def publish(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            raise RuntimeError("bucket unavailable")

    context = SimpleNamespace(
        resolve_subagent_model=lambda *args, **kwargs: (
            "anthropic:claude-test",
            "anthropic:claude-test",
            "anthropic",
        ),
        artifact_store=FailingStore(),
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
        assert returned.metadata["artifact_error"] == (
            "Artifacts could not be published."
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


def test_managed_schema_exposes_attachment_refs_only_with_a_resolver() -> None:
    without_resolver = Notebook(cast(Any, SimpleNamespace()))
    without_schema = (
        without_resolver.get_toolset().tools["analyze_data"].function_schema.json_schema
    )
    assert list(without_schema["properties"]) == ["goal", "files"]

    with_resolver = Notebook(
        cast(Any, SimpleNamespace(workspace_input_resolver=object()))
    )
    with_schema = (
        with_resolver.get_toolset().tools["analyze_data"].function_schema.json_schema
    )
    assert list(with_schema["properties"]) == [
        "goal",
        "files",
        "attachment_refs",
    ]
    assert with_schema["properties"]["attachment_refs"]["description"].startswith(
        "Optional opaque"
    )


def test_installed_capability_is_always_registered() -> None:
    notebook = capability_module.capability(cast(Any, SimpleNamespace()))
    assert isinstance(notebook, Notebook)
    assert notebook.tool.name == "analyze_data"
    assert notebook.get_toolset().tools["analyze_data"].sequential is True
