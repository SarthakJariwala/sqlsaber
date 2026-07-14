"""SQLsaber-independent notebook analysis inputs and outputs."""

from __future__ import annotations

from dataclasses import dataclass

from .execution import NotebookInput


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    file: str
    sql: str | None = None
    source: str | None = None


@dataclass(frozen=True, slots=True)
class Workspace:
    files: tuple[NotebookInput, ...]
    manifest: tuple[ManifestEntry, ...] = ()

    @classmethod
    def from_files(
        cls,
        files: list[tuple[str, bytes]] | tuple[tuple[str, bytes], ...],
        *,
        source: str | None = None,
    ) -> Workspace:
        return cls(
            files=tuple(NotebookInput(name, data) for name, data in files),
            manifest=tuple(ManifestEntry(name, source=source) for name, _ in files),
        )


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
