"""SQLsaber-independent notebook analysis inputs and outputs."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

from .execution import NotebookInput


@dataclass(frozen=True, slots=True)
class WorkspaceFile:
    """One trusted, provider-neutral file supplied to a notebook workspace."""

    name: str
    data: bytes
    media_type: str | None = None
    provenance: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("WorkspaceFile.name must be a string")
        if not isinstance(self.data, bytes):
            raise TypeError("WorkspaceFile.data must be bytes")
        if self.media_type is not None and not isinstance(self.media_type, str):
            raise TypeError("WorkspaceFile.media_type must be a string or None")
        if not isinstance(self.provenance, Mapping) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in self.provenance.items()
        ):
            raise TypeError("WorkspaceFile.provenance must map strings to strings")
        object.__setattr__(
            self,
            "provenance",
            MappingProxyType(dict(self.provenance)),
        )


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    file: str
    sql: str | None = None
    source: str | None = None
    media_type: str | None = None
    provenance: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "provenance",
            MappingProxyType(dict(self.provenance)),
        )


@dataclass(frozen=True, slots=True)
class Workspace:
    files: tuple[NotebookInput, ...]
    manifest: tuple[ManifestEntry, ...] = ()

    @classmethod
    def from_files(
        cls,
        files: Sequence[tuple[str, bytes] | WorkspaceFile],
        *,
        source: str | None = None,
    ) -> Workspace:
        inputs: list[NotebookInput] = []
        manifest: list[ManifestEntry] = []
        for item in files:
            if isinstance(item, WorkspaceFile):
                inputs.append(NotebookInput(item.name, item.data))
                manifest.append(
                    ManifestEntry(
                        item.name,
                        media_type=item.media_type,
                        provenance=item.provenance,
                    )
                )
            else:
                name, data = item
                inputs.append(NotebookInput(name, data))
                manifest.append(ManifestEntry(name, source=source))
        return cls(files=tuple(inputs), manifest=tuple(manifest))


def workspace_manifest_bytes(workspace: Workspace) -> bytes:
    """Serialize adapter-neutral metadata for files staged under ``../inputs``."""

    entries = {entry.file: entry for entry in workspace.manifest}
    manifest = []
    for item in workspace.files:
        entry = entries.get(item.name)
        manifest.append(
            {
                "file": f"../inputs/{item.name}",
                "sql": entry.sql if entry is not None else None,
                "source": entry.source if entry is not None else None,
                "media_type": entry.media_type if entry is not None else None,
                "provenance": dict(entry.provenance) if entry is not None else {},
            }
        )
    return json.dumps(manifest, indent=2, sort_keys=True).encode()


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    name: str
    data: bytes
    media_type: str


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    answer: str
    notebook: bytes
    images: list[bytes]
    files: list[ArtifactRef]
    provenance: list[str]
