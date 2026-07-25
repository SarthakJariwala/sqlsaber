"""Resolve durable artifact references from Pydantic AI message history."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from pydantic_ai.messages import ModelMessage, ToolReturnPart

from sqlsaber.artifacts import (
    ArtifactContext,
    ArtifactStore,
    ArtifactUnavailable,
    LoadedArtifact,
    StoredArtifact,
    artifact_publication_from_metadata,
    validate_loaded_artifact,
)


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    """One publication reference retained by a tool-return message."""

    tool_call_id: str
    publication_id: str
    publication_kind: str
    artifacts: tuple[StoredArtifact, ...]


@dataclass(frozen=True, slots=True)
class ResolvedArtifactPublication:
    """Available and unavailable members of one artifact publication."""

    reference: ArtifactReference
    artifacts: tuple[LoadedArtifact, ...]
    unavailable: tuple[StoredArtifact, ...]


def artifact_context_from_run(ctx: object) -> ArtifactContext:
    """Build artifact context from the current Pydantic AI run context."""

    metadata = getattr(ctx, "metadata", None)
    return ArtifactContext(
        run_id=getattr(ctx, "run_id", None),
        conversation_id=getattr(ctx, "conversation_id", None),
        tool_call_id=getattr(ctx, "tool_call_id", None),
        metadata=metadata if isinstance(metadata, Mapping) else {},
    )


def artifact_references_from_messages(
    messages: Sequence[ModelMessage],
) -> list[ArtifactReference]:
    """Extract ordered, unique publication references from message history."""

    references: list[ArtifactReference] = []
    seen: dict[str, tuple[str, tuple[StoredArtifact, ...]]] = {}
    for message in messages:
        for part in message.parts:
            if not isinstance(part, ToolReturnPart):
                continue
            publication = artifact_publication_from_metadata(part.metadata)
            if publication is None:
                if isinstance(part.metadata, Mapping) and (
                    "artifact_publication" in part.metadata
                ):
                    raise ArtifactUnavailable()
                continue
            signature = (publication.kind, publication.artifacts)
            previous = seen.get(publication.id)
            if previous is not None:
                if previous != signature:
                    raise ArtifactUnavailable()
                continue
            seen[publication.id] = signature
            references.append(
                ArtifactReference(
                    tool_call_id=part.tool_call_id,
                    publication_id=publication.id,
                    publication_kind=publication.kind,
                    artifacts=publication.artifacts,
                )
            )
    return references


async def resolve_artifact_publication(
    reference: ArtifactReference,
    *,
    store: ArtifactStore,
    context: ArtifactContext,
) -> ResolvedArtifactPublication:
    """Retrieve and verify each referenced publication member independently."""

    available: list[LoadedArtifact] = []
    unavailable: list[StoredArtifact] = []
    for expected in reference.artifacts:
        try:
            loaded = await store.get(expected.id, context=context)
            available.append(validate_loaded_artifact(loaded, expected=expected))
        except ArtifactUnavailable:
            unavailable.append(expected)
    return ResolvedArtifactPublication(
        reference=reference,
        artifacts=tuple(available),
        unavailable=tuple(unavailable),
    )
