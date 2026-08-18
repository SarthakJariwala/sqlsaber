"""Application-owned resolution contract for managed notebook inputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class WorkspaceResolutionContext:
    """Minimal run and authorization scope supplied to an input resolver."""

    run_id: str | None = None
    conversation_id: str | None = None
    tool_call_id: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@runtime_checkable
class WorkspaceInputFile(Protocol):
    """Structural input value returned by a workspace input resolver."""

    @property
    def name(self) -> str: ...

    @property
    def data(self) -> bytes: ...

    @property
    def media_type(self) -> str | None: ...

    @property
    def provenance(self) -> Mapping[str, str]: ...


@runtime_checkable
class WorkspaceInputResolver(Protocol):
    """Resolve opaque references using application-owned authorization.

    Implementations must authorize every reference against the current context.
    Possession is not authorization, and references must never be interpreted as
    model-selected filesystem paths, URLs, bucket names, or object keys.
    """

    async def resolve(
        self,
        refs: Sequence[str],
        *,
        context: WorkspaceResolutionContext,
    ) -> Sequence[WorkspaceInputFile]: ...
