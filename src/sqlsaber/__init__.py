"""SQLSaber CLI - Agentic SQL assistant like Claude Code but for SQL."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .api import SQLSaber
    from .artifacts import (
        Artifact,
        ArtifactBundle,
        ArtifactContext,
        ArtifactPublication,
        ArtifactPublisher,
        FilesystemArtifactPublisher,
        InMemoryArtifactPublisher,
        StoredArtifact,
    )
    from .capabilities import Knowledge, SqlTools
    from .options import SQLSaberOptions
    from .overrides import ModelOverides
    from .query_results import (
        FilesystemQueryResultStore,
        InMemoryQueryResultStore,
        LoadedQueryResult,
        QueryResultContext,
        QueryResultData,
        QueryResultId,
        QueryResultStore,
        QueryResultUnavailable,
        StoredQueryResult,
    )

__all__ = [
    "Artifact",
    "ArtifactBundle",
    "ArtifactContext",
    "ArtifactPublication",
    "ArtifactPublisher",
    "FilesystemArtifactPublisher",
    "InMemoryArtifactPublisher",
    "FilesystemQueryResultStore",
    "InMemoryQueryResultStore",
    "Knowledge",
    "LoadedQueryResult",
    "ModelOverides",
    "QueryResultContext",
    "QueryResultData",
    "QueryResultId",
    "QueryResultStore",
    "QueryResultUnavailable",
    "SQLSaber",
    "SQLSaberOptions",
    "SqlTools",
    "StoredArtifact",
    "StoredQueryResult",
]


def __getattr__(name: str):
    """Lazy import for SQLSaber to avoid heavy startup imports."""
    if name == "SQLSaber":
        from .api import SQLSaber

        return SQLSaber
    if name == "SQLSaberOptions":
        from .options import SQLSaberOptions

        return SQLSaberOptions
    if name in {
        "Artifact",
        "ArtifactBundle",
        "ArtifactContext",
        "ArtifactPublication",
        "ArtifactPublisher",
        "FilesystemArtifactPublisher",
        "InMemoryArtifactPublisher",
        "StoredArtifact",
    }:
        from . import artifacts

        return getattr(artifacts, name)
    if name in {
        "FilesystemQueryResultStore",
        "InMemoryQueryResultStore",
        "LoadedQueryResult",
        "QueryResultContext",
        "QueryResultData",
        "QueryResultId",
        "QueryResultStore",
        "QueryResultUnavailable",
        "StoredQueryResult",
    }:
        from . import query_results

        return getattr(query_results, name)
    if name == "ModelOverides":
        from .overrides import ModelOverides

        return ModelOverides
    if name == "SqlTools":
        from .capabilities import SqlTools

        return SqlTools
    if name == "Knowledge":
        from .capabilities import Knowledge

        return Knowledge
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
