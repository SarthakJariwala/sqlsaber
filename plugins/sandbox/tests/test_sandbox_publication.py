"""Sandbox result publication through core artifact stores."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from sqlsaber.artifacts import (
    ArtifactBundle,
    ArtifactContext,
    ArtifactPublicationError,
    InMemoryArtifactStore,
)
from sqlsaber_sandbox.publication import publish_analysis
from sqlsaber_sandbox.result import AnalysisResult, ArtifactRef, CellResult


def _result() -> AnalysisResult:
    return AnalysisResult(
        session_id="session-1",
        analysis_id="analysis-1",
        answer="Done.",
        cells=(
            CellResult(
                execution_id="execution-1",
                status="ok",
                outputs=({"output_type": "stream", "text": "done\n"},),
                execution_count=1,
            ),
        ),
        files=(
            ArtifactRef("reports/model.bin", b"\x00\xff\x10binary"),
            ArtifactRef("plots/chart.png", b"\x89PNG\r\n", "image/png"),
        ),
        notebook=b"\x00{notebook}\xff",
    )


@pytest.mark.asyncio
async def test_publication_preserves_binary_bytes_and_artifact_shape() -> None:
    result = _result()
    store = InMemoryArtifactStore()
    context = ArtifactContext(conversation_id="conversation-1")

    publication = await publish_analysis(result, store=store, context=context)

    assert publication.kind == "sandbox-analysis"
    assert [
        (item.name, item.media_type, item.kind) for item in publication.artifacts
    ] == [
        ("analysis.ipynb", "application/x-ipynb+json", "notebook"),
        ("files/reports/model.bin", "application/octet-stream", "file"),
        ("files/plots/chart.png", "image/png", "image"),
    ]
    loaded = [
        await store.get(item.id, context=context) for item in publication.artifacts
    ]
    assert [item.data for item in loaded] == [
        result.notebook,
        result.files[0].data,
        result.files[1].data,
    ]


@pytest.mark.asyncio
async def test_completed_snapshot_can_publish_repeatedly_without_live_state() -> None:
    result = _result()
    store = InMemoryArtifactStore()

    first = await publish_analysis(result, store=store, context=ArtifactContext())
    second = await publish_analysis(result, store=store, context=ArtifactContext())

    assert first.id != second.id
    assert [item.sha256 for item in first.artifacts] == [
        item.sha256 for item in second.artifacts
    ]
    with pytest.raises(FrozenInstanceError):
        result.answer = "changed"  # ty: ignore[invalid-assignment]
    with pytest.raises(FrozenInstanceError):
        result.cells[0].status = "changed"  # ty: ignore[invalid-assignment]


@pytest.mark.asyncio
async def test_publication_forwards_context_and_immutable_bundle() -> None:
    class RecordingStore(InMemoryArtifactStore):
        bundle: ArtifactBundle | None = None
        context: ArtifactContext | None = None

        async def publish(self, bundle, *, context):
            self.bundle = bundle
            self.context = context
            return await super().publish(bundle, context=context)

    store = RecordingStore()
    context = ArtifactContext(
        run_id="run-1",
        conversation_id="conversation-1",
        metadata={"tenant_id": "acme"},
    )

    await publish_analysis(_result(), store=store, context=context)

    assert store.context is context
    assert store.bundle is not None
    assert store.bundle.kind == "sandbox-analysis"
    assert isinstance(store.bundle.artifacts, tuple)


@pytest.mark.asyncio
async def test_publication_failure_is_propagated() -> None:
    class FailingStore(InMemoryArtifactStore):
        async def publish(self, bundle, *, context):
            del bundle, context
            raise ArtifactPublicationError("storage unavailable")

    with pytest.raises(ArtifactPublicationError, match="storage unavailable"):
        await publish_analysis(
            _result(),
            store=FailingStore(),
            context=ArtifactContext(),
        )


@pytest.mark.parametrize(
    "name",
    ["", "/absolute", "../escape", "nested/../escape", "a//b", "a\\b", "bad\n"],
)
def test_artifact_refs_require_safe_relative_nested_paths(name: str) -> None:
    with pytest.raises(ValueError, match="Unsafe artifact path"):
        ArtifactRef(name, b"data")

    assert ArtifactRef("nested/report.csv", b"data").name == "nested/report.csv"
