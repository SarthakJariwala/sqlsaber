"""Cloud-neutral artifacts produced by SQLsaber capabilities."""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Protocol, cast, runtime_checkable

ArtifactFailureMode = Literal["required", "best_effort"]
ArtifactKind = Literal["notebook", "image", "file"]
_FILESYSTEM_ARTIFACTS_DIRECTORY = "artifacts"
_FILESYSTEM_MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True, slots=True)
class Artifact:
    """One bounded artifact ready to be persisted outside an agent run."""

    name: str
    data: bytes
    media_type: str
    kind: ArtifactKind = "file"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_relative_name(self.name)
        if not isinstance(self.data, bytes):
            raise TypeError("Artifact.data must be bytes")
        if not self.media_type.strip():
            raise ValueError("Artifact.media_type cannot be empty")

    @property
    def size(self) -> int:
        return len(self.data)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()


@dataclass(frozen=True, slots=True)
class ArtifactBundle:
    """A related collection of artifacts from one capability operation."""

    kind: str
    artifacts: tuple[Artifact, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.kind.strip():
            raise ValueError("ArtifactBundle.kind cannot be empty")
        names = [artifact.name for artifact in self.artifacts]
        if len(names) != len(set(names)):
            raise ValueError("Artifact names must be unique within a bundle")


@dataclass(frozen=True, slots=True)
class ArtifactContext:
    """Run identity available to an artifact publisher for safe namespacing."""

    run_id: str | None = None
    conversation_id: str | None = None
    tool_call_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StoredArtifact:
    """A durable, serializable reference returned by an artifact publisher."""

    id: str
    name: str
    media_type: str
    size: int
    sha256: str
    uri: str
    kind: ArtifactKind = "file"

    def to_dict(self) -> dict[str, str | int]:
        return {
            "id": self.id,
            "name": self.name,
            "media_type": self.media_type,
            "size": self.size,
            "sha256": self.sha256,
            "uri": self.uri,
            "kind": self.kind,
        }

    @classmethod
    def from_dict(cls, value: object) -> StoredArtifact | None:
        if not isinstance(value, Mapping):
            return None
        mapping = cast(Mapping[str, Any], value)
        try:
            kind = mapping.get("kind", "file")
            if kind not in ("notebook", "image", "file"):
                return None
            return cls(
                id=str(mapping["id"]),
                name=str(mapping["name"]),
                media_type=str(mapping["media_type"]),
                size=int(mapping["size"]),
                sha256=str(mapping["sha256"]),
                uri=str(mapping["uri"]),
                kind=kind,
            )
        except (KeyError, TypeError, ValueError):
            return None


@dataclass(frozen=True, slots=True)
class ArtifactPublication:
    """Durable result of publishing one artifact bundle."""

    id: str
    artifacts: tuple[StoredArtifact, ...]

    def to_metadata(self) -> dict[str, object]:
        return {
            "analysis_id": self.id,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
        }


@runtime_checkable
class ArtifactPublisher(Protocol):
    """Application-owned persistence boundary for capability artifacts."""

    async def publish(
        self,
        bundle: ArtifactBundle,
        *,
        context: ArtifactContext,
    ) -> ArtifactPublication: ...


class InMemoryArtifactPublisher:
    """Artifact publisher for tests and short-lived programmatic workflows."""

    def __init__(self) -> None:
        self.publications: dict[str, tuple[ArtifactBundle, ArtifactContext]] = {}

    async def publish(
        self,
        bundle: ArtifactBundle,
        *,
        context: ArtifactContext,
    ) -> ArtifactPublication:
        publication_id = uuid.uuid4().hex
        self.publications[publication_id] = (bundle, context)
        artifacts = tuple(
            StoredArtifact(
                id=f"{publication_id}:{artifact.name}",
                name=artifact.name,
                media_type=artifact.media_type,
                size=artifact.size,
                sha256=artifact.sha256,
                uri=f"memory://{publication_id}/{artifact.name}",
                kind=artifact.kind,
            )
            for artifact in bundle.artifacts
        )
        return ArtifactPublication(publication_id, artifacts)


class FilesystemArtifactPublisher:
    """Atomically persist metadata and artifacts in separate filesystem namespaces."""

    def __init__(
        self,
        root: str | Path,
        *,
        namespace: Callable[[ArtifactContext], str | None] | None = None,
    ) -> None:
        self.root = Path(root)
        self.namespace = namespace

    async def publish(
        self,
        bundle: ArtifactBundle,
        *,
        context: ArtifactContext,
    ) -> ArtifactPublication:
        return await asyncio.to_thread(self._publish_sync, bundle, context)

    def _publish_sync(
        self,
        bundle: ArtifactBundle,
        context: ArtifactContext,
    ) -> ArtifactPublication:
        publication_id = uuid.uuid4().hex
        relative_parent = PurePosixPath()
        if self.namespace is not None:
            selected = self.namespace(context)
            if selected:
                _validate_relative_name(selected)
                relative_parent = PurePosixPath(selected)

        parent = self.root.joinpath(*relative_parent.parts)
        target = parent / publication_id
        temporary = parent / f".{publication_id}.tmp"
        parent.mkdir(parents=True, exist_ok=True)
        temporary.mkdir(mode=0o700)
        artifact_root = temporary / _FILESYSTEM_ARTIFACTS_DIRECTORY
        stored: list[StoredArtifact] = []
        try:
            artifact_root.mkdir(mode=0o700)
            for artifact in bundle.artifacts:
                relative = PurePosixPath(artifact.name)
                path = artifact_root.joinpath(*relative.parts)
                path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                path.write_bytes(artifact.data)
                path.chmod(0o600)
                final_path = target.joinpath(
                    _FILESYSTEM_ARTIFACTS_DIRECTORY, *relative.parts
                ).resolve()
                stored.append(
                    StoredArtifact(
                        id=f"{publication_id}:{artifact.name}",
                        name=artifact.name,
                        media_type=artifact.media_type,
                        size=artifact.size,
                        sha256=artifact.sha256,
                        uri=final_path.as_uri(),
                        kind=artifact.kind,
                    )
                )
            manifest = {
                "id": publication_id,
                "kind": bundle.kind,
                "run_id": context.run_id,
                "conversation_id": context.conversation_id,
                "tool_call_id": context.tool_call_id,
                "metadata": _json_safe(bundle.metadata),
                "artifacts": [artifact.to_dict() for artifact in stored],
            }
            manifest_path = temporary / _FILESYSTEM_MANIFEST_NAME
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
            manifest_path.chmod(0o600)
            temporary.replace(target)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return ArtifactPublication(publication_id, tuple(stored))


def artifacts_from_metadata(metadata: object) -> list[StoredArtifact]:
    """Parse durable artifact references from tool-return metadata."""

    if not isinstance(metadata, Mapping):
        return []
    mapping = cast(Mapping[str, Any], metadata)
    values = mapping.get("artifacts")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return []
    parsed: list[StoredArtifact] = []
    for value in values:
        artifact = StoredArtifact.from_dict(value)
        if artifact is not None:
            parsed.append(artifact)
    return parsed


def _validate_relative_name(name: str) -> None:
    path = PurePosixPath(name)
    if (
        not name
        or name == "."
        or path.is_absolute()
        or path.as_posix() != name
        or any(part in ("", ".", "..") for part in path.parts)
        or "\\" in name
        or "\x00" in name
    ):
        raise ValueError(f"Unsafe artifact path: {name!r}")


def _json_safe(value: object) -> object:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return str(value)
    return value
