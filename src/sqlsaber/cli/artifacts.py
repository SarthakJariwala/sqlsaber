"""CLI artifact-store construction and hydration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import TypeGuard

import platformdirs
from pydantic_ai.messages import ModelMessage

from sqlsaber.artifact_resolution import (
    ResolvedArtifactPublication,
    artifact_references_from_messages,
    resolve_artifact_publication,
)
from sqlsaber.artifacts import (
    ArtifactContext,
    ArtifactStore,
    FilesystemArtifactStore,
)


async def resolve_cli_artifact_publications(
    messages: list[ModelMessage],
    *,
    store: ArtifactStore,
    context: ArtifactContext | None = None,
) -> dict[str, ResolvedArtifactPublication]:
    """Resolve every retained publication for read-only CLI replay."""

    resolved: dict[str, ResolvedArtifactPublication] = {}
    for reference in artifact_references_from_messages(messages):
        resolved[reference.publication_id] = await resolve_artifact_publication(
            reference,
            store=store,
            context=context or ArtifactContext(),
        )
    return resolved


async def unavailable_artifact_ids(
    messages: list[ModelMessage],
    *,
    store: ArtifactStore,
    context: ArtifactContext | None = None,
) -> set[str]:
    """Return referenced artifact IDs that cannot be safely retrieved."""

    publications = await resolve_cli_artifact_publications(
        messages,
        store=store,
        context=context,
    )
    return {
        artifact.id
        for publication in publications.values()
        for artifact in publication.unavailable
    }


class _CLIArtifactStore(FilesystemArtifactStore):
    """Marker type for the filesystem store whose retention the CLI owns."""


def is_cli_artifact_store(store: object) -> TypeGuard[_CLIArtifactStore]:
    """Return whether artifact retention belongs to the SQLsaber CLI."""

    return isinstance(store, _CLIArtifactStore)


def cli_artifact_store() -> FilesystemArtifactStore:
    """Return the persistent artifact store used by CLI execution and replay."""

    return _CLIArtifactStore(Path(platformdirs.user_data_dir("sqlsaber")) / "artifacts")
