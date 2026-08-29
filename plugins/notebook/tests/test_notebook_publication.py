"""Public notebook artifact publication and replay behavior."""

from __future__ import annotations

import base64
import io
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import url2pathname

import nbformat
import pytest
from PIL import Image
from pydantic_ai.messages import (
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from rich.console import Console

from sqlsaber import ArtifactContext, FilesystemArtifactStore, InMemoryArtifactStore
from sqlsaber.artifact_resolution import (
    ArtifactReference,
    resolve_artifact_publication,
)
from sqlsaber.artifacts import ArtifactBundle, ArtifactUnavailable
from sqlsaber.cli.artifacts import resolve_cli_artifact_publications
from sqlsaber.cli.display import DisplayManager
from sqlsaber.cli.threads import _render_transcript
from sqlsaber.threads import ThreadStorage
from sqlsaber_notebook import AnalysisResult, ArtifactRef, publish_analysis
from sqlsaber_notebook.capability import AnalyzeDataTool
from sqlsaber_notebook.publication import display_from_publication


def test_notebook_sdk_exports_are_lazy_and_supported() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import sqlsaber_notebook
assert 'sqlsaber_notebook.analyst' not in sys.modules
assert 'sqlsaber_notebook.publication' not in sys.modules
from sqlsaber_notebook import (
    AnalysisResult, ArtifactRef, ManifestEntry, Workspace, WorkspaceFile,
    WorkspaceInputResolver, WorkspaceResolutionContext, analyze, publish_analysis,
)
assert all((
    AnalysisResult, ArtifactRef, ManifestEntry, Workspace, WorkspaceFile,
    WorkspaceInputResolver, WorkspaceResolutionContext, analyze, publish_analysis,
))
""",
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def _png_bytes(color: str) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.asyncio
async def test_embedded_analysis_publishes_retrievable_notebook() -> None:
    notebook = b'{"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}'
    result = AnalysisResult(
        answer="Done.",
        notebook=notebook,
        images=[],
        files=[],
        provenance=["cell:0"],
    )
    store = InMemoryArtifactStore()

    publication = await publish_analysis(
        result,
        store=store,
        context=ArtifactContext(conversation_id="conversation-1"),
    )

    assert publication.kind == "notebook-analysis"
    assert len(publication.artifacts) == 1
    descriptor = publication.artifacts[0]
    assert descriptor.name == "analysis.ipynb"
    assert descriptor.kind == "notebook"
    assert descriptor.media_type == "application/x-ipynb+json"
    loaded = await store.get(
        descriptor.id,
        context=ArtifactContext(conversation_id="conversation-1"),
    )
    assert loaded.data == notebook


@pytest.mark.asyncio
async def test_embedded_analysis_publishes_plots_and_nested_generated_files() -> None:
    result = AnalysisResult(
        answer="Done.",
        notebook=b"notebook",
        images=[b"png"],
        files=[ArtifactRef("reports/evidence.csv", b"a,b\n1,2\n", "text/csv")],
        provenance=[],
    )
    store = InMemoryArtifactStore()

    publication = await publish_analysis(
        result,
        store=store,
        context=ArtifactContext(),
    )

    assert [
        (artifact.name, artifact.kind, artifact.media_type)
        for artifact in publication.artifacts
    ] == [
        ("analysis.ipynb", "notebook", "application/x-ipynb+json"),
        ("plots/plot_1.png", "image", "image/png"),
        ("files/reports/evidence.csv", "file", "text/csv"),
    ]
    loaded = [
        await store.get(artifact.id, context=ArtifactContext())
        for artifact in publication.artifacts
    ]
    assert [artifact.data for artifact in loaded] == [
        b"notebook",
        b"png",
        b"a,b\n1,2\n",
    ]


@pytest.mark.asyncio
async def test_embedded_publication_carries_analysis_provenance_only() -> None:
    class RecordingStore(InMemoryArtifactStore):
        published_bundle: ArtifactBundle | None = None

        async def publish(
            self,
            bundle: ArtifactBundle,
            *,
            context: ArtifactContext,
        ):
            self.published_bundle = bundle
            return await super().publish(bundle, context=context)

    store = RecordingStore()
    result = AnalysisResult(
        answer="Done.",
        notebook=b"notebook",
        images=[],
        files=[],
        provenance=["input:sales.json", "cell:0"],
    )

    await publish_analysis(result, store=store, context=ArtifactContext())

    assert store.published_bundle is not None
    assert store.published_bundle.metadata == {
        "provenance": ["input:sales.json", "cell:0"]
    }


@pytest.mark.asyncio
async def test_embedded_publication_forwards_current_cloud_context_unchanged() -> None:
    class TenantAwareStore(InMemoryArtifactStore):
        received_context: ArtifactContext | None = None

        async def publish(
            self,
            bundle: ArtifactBundle,
            *,
            context: ArtifactContext,
        ):
            self.received_context = context
            return await super().publish(bundle, context=context)

    store = TenantAwareStore()
    context = ArtifactContext(
        run_id="run-1",
        conversation_id="conversation-1",
        tool_call_id="tool-1",
        metadata={"tenant_id": "acme", "user_id": "user-1"},
    )
    result = AnalysisResult("Done.", b"notebook", [], [], ["cell:0"])

    await publish_analysis(result, store=store, context=context)

    assert store.received_context is context


@pytest.mark.asyncio
async def test_published_analysis_reconstructs_notebook_after_store_restart(
    tmp_path,
) -> None:
    notebook = nbformat.v4.new_notebook()
    notebook.cells = [
        nbformat.v4.new_code_cell(
            "print('durable evidence')",
            outputs=[
                nbformat.v4.new_output(
                    "stream", name="stdout", text="durable evidence\n"
                )
            ],
        )
    ]
    store = FilesystemArtifactStore(tmp_path / "artifacts")
    publication = await publish_analysis(
        AnalysisResult(
            answer="Done.",
            notebook=nbformat.writes(notebook).encode(),
            images=[],
            files=[],
            provenance=["cell:0"],
        ),
        store=store,
        context=ArtifactContext(),
    )
    reference = ArtifactReference(
        tool_call_id="tool-1",
        publication_id=publication.id,
        publication_kind=publication.kind,
        artifacts=publication.artifacts,
    )

    resolved = await resolve_artifact_publication(
        reference,
        store=FilesystemArtifactStore(tmp_path / "artifacts"),
        context=ArtifactContext(),
    )
    display = display_from_publication(resolved)

    assert "print('durable evidence')" in display.markdown
    assert "durable evidence" in display.markdown
    assert display.images == ()
    assert display.files == ()


@pytest.mark.asyncio
async def test_published_display_deduplicates_images_and_lists_generated_files(
    tmp_path,
) -> None:
    duplicate = _png_bytes("red")
    distinct = _png_bytes("blue")
    notebook = nbformat.v4.new_notebook()
    notebook.cells = [
        nbformat.v4.new_code_cell(
            "plot()",
            outputs=[
                nbformat.v4.new_output(
                    "display_data",
                    data={
                        "image/png": base64.b64encode(duplicate).decode(),
                        "text/plain": "<plot>",
                    },
                )
            ],
        )
    ]
    store = FilesystemArtifactStore(tmp_path / "artifacts")
    publication = await publish_analysis(
        AnalysisResult(
            answer="Done.",
            notebook=nbformat.writes(notebook).encode(),
            images=[duplicate, distinct],
            files=[ArtifactRef("reports/evidence.csv", b"a,b\n", "text/csv")],
            provenance=["cell:0"],
        ),
        store=store,
        context=ArtifactContext(),
    )
    resolved = await resolve_artifact_publication(
        ArtifactReference(
            tool_call_id="tool-1",
            publication_id=publication.id,
            publication_kind=publication.kind,
            artifacts=publication.artifacts,
        ),
        store=FilesystemArtifactStore(tmp_path / "artifacts"),
        context=ArtifactContext(),
    )

    display = display_from_publication(resolved)

    assert display.images == (duplicate, distinct)
    assert [artifact.name for artifact in display.files] == [
        "files/reports/evidence.csv"
    ]


@pytest.mark.asyncio
async def test_fresh_notebook_renderer_uses_resolved_publication(tmp_path) -> None:
    notebook = nbformat.v4.new_notebook()
    notebook.cells = [nbformat.v4.new_code_cell("print('replayed notebook')")]
    store = FilesystemArtifactStore(tmp_path / "artifacts")
    publication = await publish_analysis(
        AnalysisResult(
            answer="Persisted answer.",
            notebook=nbformat.writes(notebook).encode(),
            images=[],
            files=[],
            provenance=["cell:0"],
        ),
        store=store,
        context=ArtifactContext(),
    )
    resolved = await resolve_artifact_publication(
        ArtifactReference(
            tool_call_id="tool-1",
            publication_id=publication.id,
            publication_kind=publication.kind,
            artifacts=publication.artifacts,
        ),
        store=FilesystemArtifactStore(tmp_path / "artifacts"),
        context=ArtifactContext(),
    )
    renderer = AnalyzeDataTool(object())
    from io import StringIO

    from sqlsaber.theme.manager import create_console

    buffer = StringIO()
    console = create_console(file=buffer, width=80, legacy_windows=False)
    display_manager = DisplayManager(console, {"analyze_data": renderer})
    display_manager.set_resolved_artifact_publications({publication.id: resolved})

    display_manager.show_tool_result(
        "analyze_data",
        "Persisted answer.",
        tool_call_id="tool-1",
        metadata=publication.to_metadata(),
    )

    rendered = buffer.getvalue()
    assert "replayed notebook" in rendered
    assert "Persisted answer" in rendered


@pytest.mark.asyncio
async def test_invalid_persisted_notebook_falls_back_to_generic_artifacts(
    tmp_path,
) -> None:
    store = FilesystemArtifactStore(tmp_path / "artifacts")
    publication = await publish_analysis(
        AnalysisResult("Answer survives.", b"not a notebook", [], [], []),
        store=store,
        context=ArtifactContext(),
    )
    resolved = await resolve_artifact_publication(
        ArtifactReference(
            tool_call_id="tool-1",
            publication_id=publication.id,
            publication_kind=publication.kind,
            artifacts=publication.artifacts,
        ),
        store=FilesystemArtifactStore(tmp_path / "artifacts"),
        context=ArtifactContext(),
    )
    renderer = AnalyzeDataTool(object())
    from io import StringIO

    from sqlsaber.theme.manager import create_console

    buffer = StringIO()
    console = create_console(file=buffer, width=80, legacy_windows=False)
    display = DisplayManager(console, {"analyze_data": renderer})
    display.set_resolved_artifact_publications({publication.id: resolved})

    display.show_tool_result(
        "analyze_data",
        "Answer survives.",
        tool_call_id="tool-1",
        metadata=publication.to_metadata(),
    )

    rendered = buffer.getvalue()
    assert "Answer survives" in rendered
    assert "could not be reconstructed" in rendered
    assert "analysis.ipynb" in rendered


@pytest.mark.asyncio
async def test_missing_persisted_notebook_is_reported_unavailable(tmp_path) -> None:
    notebook = nbformat.v4.new_notebook()
    publication = await publish_analysis(
        AnalysisResult("Answer.", nbformat.writes(notebook).encode(), [], [], []),
        store=FilesystemArtifactStore(tmp_path / "artifacts"),
        context=ArtifactContext(),
    )
    notebook_descriptor = publication.artifacts[0]
    Path(url2pathname(urlsplit(notebook_descriptor.uri).path)).unlink()
    resolved = await resolve_artifact_publication(
        ArtifactReference(
            tool_call_id="tool-1",
            publication_id=publication.id,
            publication_kind=publication.kind,
            artifacts=publication.artifacts,
        ),
        store=FilesystemArtifactStore(tmp_path / "artifacts"),
        context=ArtifactContext(),
    )

    with pytest.raises(ArtifactUnavailable):
        display_from_publication(resolved)

    console = Console(record=True, force_terminal=False, width=100)
    display = DisplayManager(
        console,
        {"analyze_data": AnalyzeDataTool(object())},
    )
    display.set_resolved_artifact_publications({publication.id: resolved})
    display.set_unavailable_artifacts({notebook_descriptor.id})
    display.show_tool_result(
        "analyze_data",
        "Answer.",
        tool_call_id="tool-1",
        metadata=publication.to_metadata(),
    )

    rendered = console.export_text()
    assert "could not be reconstructed" in rendered
    assert "analysis.ipynb" in rendered
    assert "unavailable" in rendered


@pytest.mark.asyncio
async def test_retained_thread_replays_notebook_and_artifact_locations(
    tmp_path,
) -> None:
    notebook = nbformat.v4.new_notebook()
    notebook.cells = [nbformat.v4.new_code_cell("print('thread replay')")]
    artifact_store = FilesystemArtifactStore(tmp_path / "artifacts")
    publication = await publish_analysis(
        AnalysisResult(
            "Persisted answer.",
            nbformat.writes(notebook).encode(),
            [],
            [ArtifactRef("report.csv", b"value\n1\n", "text/csv")],
            ["cell:0"],
        ),
        store=artifact_store,
        context=ArtifactContext(),
    )
    messages = [
        ModelRequest(parts=[UserPromptPart("Analyze")]),
        ModelResponse(
            parts=[
                ToolCallPart("analyze_data", {"goal": "Analyze"}, tool_call_id="tool-1")
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    "analyze_data",
                    "Persisted answer.",
                    tool_call_id="tool-1",
                    metadata=publication.to_metadata(),
                )
            ]
        ),
    ]
    threads = ThreadStorage()
    threads.db_path = tmp_path / "threads.db"
    snapshot = await threads.save_snapshot(
        messages_json=ModelMessagesTypeAdapter.dump_json(messages),
        database_name="test",
    )

    retained = await threads.get_thread_messages(snapshot)
    resolved = await resolve_cli_artifact_publications(
        retained,
        store=FilesystemArtifactStore(tmp_path / "artifacts"),
    )
    console = Console(record=True, force_terminal=False, width=120)
    _render_transcript(
        console,
        retained,
        resolved_artifacts=resolved,
    )

    rendered = console.export_text()
    assert "thread replay" in rendered
    assert "Persisted answer" in rendered
    assert "analysis.ipynb" in rendered
    assert publication.artifacts[0].uri in rendered.replace("\n", "")
    assert "files/report.csv" in rendered
