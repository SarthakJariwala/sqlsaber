"""SQLSaber CLI - Agentic SQL assistant like Claude Code but for SQL."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .artifacts import (
        Artifact,
        ArtifactBundle,
        ArtifactContext,
        ArtifactPublication,
        ArtifactPublicationError,
        ArtifactStore,
        ArtifactStoreError,
        ArtifactUnavailable,
        FilesystemArtifactStore,
        InMemoryArtifactStore,
        LoadedArtifact,
        StoredArtifact,
    )
    from .capabilities import Knowledge, SqlTools
    from .config.settings import ThinkingLevel
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
    from .sdk.client import SQLSaber, SQLSaberResult
    from .sdk.errors import (
        RunInProgressError,
        SQLSaberClosedError,
        SQLSaberError,
        ThreadDatabaseRequiredError,
        ThreadDatabaseUnavailableError,
        ThreadNotFoundError,
        ThreadResumeError,
        ThreadResumeHistoryError,
        ThreadResumeMetadataError,
    )
    from .sdk.options import SQLSaberOptions
    from .sdk.types import SQLSaberInfo, TableInfo, ThinkingState
    from .workspace_inputs import WorkspaceInputResolver, WorkspaceResolutionContext

__all__ = [
    "Artifact",
    "ArtifactBundle",
    "ArtifactContext",
    "ArtifactPublication",
    "ArtifactPublicationError",
    "ArtifactStore",
    "ArtifactStoreError",
    "ArtifactUnavailable",
    "FilesystemArtifactStore",
    "InMemoryArtifactStore",
    "LoadedArtifact",
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
    "RunInProgressError",
    "SQLSaber",
    "SQLSaberClosedError",
    "SQLSaberError",
    "SQLSaberInfo",
    "SQLSaberOptions",
    "SQLSaberResult",
    "SqlTools",
    "TableInfo",
    "ThinkingLevel",
    "ThinkingState",
    "ThreadDatabaseRequiredError",
    "ThreadDatabaseUnavailableError",
    "ThreadNotFoundError",
    "ThreadResumeError",
    "ThreadResumeHistoryError",
    "ThreadResumeMetadataError",
    "StoredArtifact",
    "StoredQueryResult",
    "WorkspaceInputResolver",
    "WorkspaceResolutionContext",
]


def __getattr__(name: str):
    """Lazy import for SQLSaber to avoid heavy startup imports."""
    if name in {"SQLSaber", "SQLSaberResult"}:
        from .sdk import client

        return getattr(client, name)
    if name == "SQLSaberOptions":
        from .sdk.options import SQLSaberOptions

        return SQLSaberOptions
    if name in {
        "Artifact",
        "ArtifactBundle",
        "ArtifactContext",
        "ArtifactPublication",
        "ArtifactPublicationError",
        "ArtifactStore",
        "ArtifactStoreError",
        "ArtifactUnavailable",
        "FilesystemArtifactStore",
        "InMemoryArtifactStore",
        "LoadedArtifact",
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
    if name == "ThinkingLevel":
        from .config.settings import ThinkingLevel

        return ThinkingLevel
    if name in {"SQLSaberInfo", "TableInfo", "ThinkingState"}:
        from .sdk import types

        return getattr(types, name)
    if name in {
        "RunInProgressError",
        "SQLSaberClosedError",
        "SQLSaberError",
        "ThreadDatabaseRequiredError",
        "ThreadDatabaseUnavailableError",
        "ThreadNotFoundError",
        "ThreadResumeError",
        "ThreadResumeHistoryError",
        "ThreadResumeMetadataError",
    }:
        from .sdk import errors

        return getattr(errors, name)
    if name in {"WorkspaceInputResolver", "WorkspaceResolutionContext"}:
        from . import workspace_inputs

        return getattr(workspace_inputs, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
