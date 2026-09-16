"""Immutable inputs and completed results for sandbox analyses."""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any

_MAX_WORKSPACE_FILENAME_BYTES = 255
_RESERVED_WORKSPACE_FILENAME = "manifest.json"


def _contains_control_characters(value: str) -> bool:
    return any(unicodedata.category(character) == "Cc" for character in value)


def validate_workspace_file_name(name: str) -> None:
    """Validate one flat filename staged in the sandbox input directory."""

    if not isinstance(name, str):
        raise TypeError("WorkspaceFile.name must be a string")
    try:
        encoded_name = name.encode("utf-8")
    except UnicodeEncodeError:
        encoded_name = b""
    if (
        not name
        or name in {".", "..", _RESERVED_WORKSPACE_FILENAME}
        or "/" in name
        or "\\" in name
        or _contains_control_characters(name)
        or not encoded_name
        or len(encoded_name) > _MAX_WORKSPACE_FILENAME_BYTES
    ):
        raise ValueError(f"Unsafe workspace filename: {name!r}")


def _validate_artifact_name(name: str) -> None:
    if not isinstance(name, str):
        raise TypeError("ArtifactRef.name must be a string")
    path = PurePosixPath(name)
    if (
        not name
        or name == "."
        or path.is_absolute()
        or path.as_posix() != name
        or any(part in {"", ".", ".."} for part in path.parts)
        or "\\" in name
        or _contains_control_characters(name)
    ):
        raise ValueError(f"Unsafe artifact path: {name!r}")


@dataclass(frozen=True, slots=True)
class WorkspaceFile:
    """One provider-neutral file admitted to a sandbox workspace."""

    name: str
    data: bytes
    media_type: str | None = None
    provenance: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_workspace_file_name(self.name)
        if not isinstance(self.data, bytes):
            raise TypeError("WorkspaceFile.data must be bytes")
        if self.media_type is not None and not isinstance(self.media_type, str):
            raise TypeError("WorkspaceFile.media_type must be a string or None")
        if not isinstance(self.provenance, Mapping):
            raise TypeError("WorkspaceFile.provenance must map strings to strings")
        try:
            provenance = dict(self.provenance)
        except Exception as exc:
            raise TypeError(
                "WorkspaceFile.provenance must map strings to strings"
            ) from exc
        if any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in provenance.items()
        ):
            raise TypeError("WorkspaceFile.provenance must map strings to strings")
        object.__setattr__(self, "provenance", MappingProxyType(provenance))


@dataclass(frozen=True, slots=True)
class Workspace:
    """A complete immutable set of files to stage for sandbox analysis."""

    files: tuple[WorkspaceFile, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.files, tuple) or any(
            not isinstance(item, WorkspaceFile) for item in self.files
        ):
            raise TypeError("Workspace.files must be a tuple of WorkspaceFile values")

    @classmethod
    def from_files(
        cls,
        files: Sequence[tuple[str, bytes] | WorkspaceFile],
    ) -> Workspace:
        workspace_files: list[WorkspaceFile] = []
        for item in files:
            if isinstance(item, WorkspaceFile):
                workspace_files.append(item)
                continue
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError(
                    "Workspace files must be WorkspaceFile values or (name, data) tuples"
                )
            name, data = item
            workspace_files.append(WorkspaceFile(name, data))
        return cls(tuple(workspace_files))

    def manifest_bytes(self) -> bytes:
        """Serialize stable metadata for files staged under ``../inputs``."""

        manifest = [
            {
                "file": f"../inputs/{item.name}",
                "media_type": item.media_type,
                "provenance": dict(item.provenance),
            }
            for item in self.files
        ]
        return json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """One completed sandbox file available for publication."""

    name: str
    data: bytes
    media_type: str = "application/octet-stream"

    def __post_init__(self) -> None:
        _validate_artifact_name(self.name)
        if not isinstance(self.data, bytes):
            raise TypeError("ArtifactRef.data must be bytes")
        if not isinstance(self.media_type, str):
            raise TypeError("ArtifactRef.media_type must be a string")
        if not self.media_type.strip():
            raise ValueError("ArtifactRef.media_type cannot be empty")


@dataclass(frozen=True, slots=True)
class CellResult:
    """One completed cell execution returned by the sandbox controller."""

    execution_id: str
    status: str
    outputs: tuple[dict[str, Any], ...]
    execution_count: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.execution_id, str):
            raise TypeError("CellResult.execution_id must be a string")
        if not isinstance(self.status, str):
            raise TypeError("CellResult.status must be a string")
        if not isinstance(self.outputs, tuple) or any(
            not isinstance(output, dict) for output in self.outputs
        ):
            raise TypeError("CellResult.outputs must be a tuple of dictionaries")
        if self.execution_count is not None and (
            isinstance(self.execution_count, bool)
            or not isinstance(self.execution_count, int)
        ):
            raise TypeError("CellResult.execution_count must be an integer or None")


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """Completed, provider-independent snapshot of one sandbox analysis."""

    session_id: str
    analysis_id: str
    answer: str
    cells: tuple[CellResult, ...]
    files: tuple[ArtifactRef, ...]
    notebook: bytes

    def __post_init__(self) -> None:
        for name in ("session_id", "analysis_id", "answer"):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"AnalysisResult.{name} must be a string")
        if not isinstance(self.cells, tuple) or any(
            not isinstance(cell, CellResult) for cell in self.cells
        ):
            raise TypeError("AnalysisResult.cells must be a tuple of CellResult values")
        if not isinstance(self.files, tuple) or any(
            not isinstance(artifact, ArtifactRef) for artifact in self.files
        ):
            raise TypeError(
                "AnalysisResult.files must be a tuple of ArtifactRef values"
            )
        if not isinstance(self.notebook, bytes):
            raise TypeError("AnalysisResult.notebook must be bytes")
