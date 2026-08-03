from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

import pytest
from pydantic_ai.messages import ModelRequest, ToolReturnPart

from sqlsaber.artifact_resolution import (
    artifact_references_from_messages,
    resolve_artifact_publication,
)
from sqlsaber.artifacts import (
    Artifact,
    ArtifactBundle,
    ArtifactContext,
    ArtifactUnavailable,
    FilesystemArtifactStore,
    InMemoryArtifactStore,
)


async def test_resolution_keeps_valid_members_when_one_is_unavailable(
    tmp_path,
) -> None:
    store = FilesystemArtifactStore(tmp_path)
    context = ArtifactContext()
    publication = await store.publish(
        ArtifactBundle(
            kind="notebook-analysis",
            artifacts=(
                Artifact("analysis.ipynb", b"notebook", "application/json"),
                Artifact("plot.png", b"png", "image/png", "image"),
            ),
        ),
        context=context,
    )
    Path(url2pathname(urlparse(publication.artifacts[1].uri).path)).write_bytes(
        b"changed"
    )
    messages = [
        ModelRequest(
            parts=[
                ToolReturnPart(
                    "analyze_data",
                    "done",
                    metadata=publication.to_metadata(),
                )
            ]
        )
    ]

    reference = artifact_references_from_messages(messages)[0]
    resolved = await resolve_artifact_publication(
        reference,
        store=store,
        context=context,
    )

    assert [loaded.data for loaded in resolved.artifacts] == [b"notebook"]
    assert resolved.unavailable == (publication.artifacts[1],)


async def test_rejects_conflicting_references_for_same_publication() -> None:
    store = InMemoryArtifactStore()
    context = ArtifactContext()
    first = await store.publish(
        ArtifactBundle(
            kind="notebook-analysis",
            artifacts=(Artifact("analysis.ipynb", b"one", "application/json"),),
        ),
        context=context,
    )
    conflicting = await store.publish(
        ArtifactBundle(
            kind="notebook-analysis",
            artifacts=(Artifact("analysis.ipynb", b"two", "application/json"),),
        ),
        context=context,
    )
    conflicting_metadata = conflicting.to_metadata()
    conflicting_metadata["artifact_publication"]["id"] = first.id  # type: ignore[index]
    messages = [
        ModelRequest(
            parts=[
                ToolReturnPart("analyze_data", "one", metadata=first.to_metadata()),
                ToolReturnPart("analyze_data", "two", metadata=conflicting_metadata),
            ]
        )
    ]

    with pytest.raises(ArtifactUnavailable):
        artifact_references_from_messages(messages)


async def test_resolves_publication_from_tool_return_metadata() -> None:
    store = InMemoryArtifactStore()
    context = ArtifactContext(conversation_id="conversation-1")
    publication = await store.publish(
        ArtifactBundle(
            kind="notebook-analysis",
            artifacts=(
                Artifact("analysis.ipynb", b"notebook", "application/json", "notebook"),
                Artifact("plots/plot.png", b"png", "image/png", "image"),
            ),
        ),
        context=context,
    )
    messages = [
        ModelRequest(
            parts=[
                ToolReturnPart(
                    "analyze_data",
                    "answer",
                    tool_call_id="tool-1",
                    metadata=publication.to_metadata(),
                )
            ]
        )
    ]

    references = artifact_references_from_messages(messages)
    resolved = await resolve_artifact_publication(
        references[0], store=store, context=context
    )

    assert resolved.reference.publication_id == publication.id
    assert [artifact.data for artifact in resolved.artifacts] == [b"notebook", b"png"]
    assert resolved.unavailable == ()
