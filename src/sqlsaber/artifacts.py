"""Cloud-neutral artifacts produced by SQLsaber capabilities."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import secrets
import shutil
import stat
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Protocol, cast, runtime_checkable
from urllib.parse import urlsplit

ArtifactFailureMode = Literal["required", "best_effort"]
ArtifactKind = Literal["notebook", "image", "file"]
_FILESYSTEM_ARTIFACTS_DIRECTORY = "artifacts"
_FILESYSTEM_MANIFEST_NAME = "manifest.json"
_ARTIFACT_SCHEMA_VERSION = 1
_PUBLICATION_ID_RE = re.compile(r"^ap_[a-f0-9]{32}$")
_ARTIFACT_ID_RE = re.compile(r"^ar_[a-f0-9]{32}$")


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
            artifact_id = mapping["id"]
            name = mapping["name"]
            media_type = mapping["media_type"]
            size = mapping["size"]
            digest = mapping["sha256"]
            uri = mapping["uri"]
            kind = mapping.get("kind", "file")
        except KeyError:
            return None
        if (
            not isinstance(artifact_id, str)
            or _ARTIFACT_ID_RE.fullmatch(artifact_id) is None
            or not isinstance(name, str)
            or not isinstance(media_type, str)
            or not media_type.strip()
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or not isinstance(digest, str)
            or re.fullmatch(r"[a-f0-9]{64}", digest) is None
            or not isinstance(uri, str)
            or not uri
            or not urlsplit(uri).scheme
            or urlsplit(uri).scheme.lower() in {"data", "javascript", "vbscript"}
            or any(ord(char) < 32 for char in uri)
            or kind not in ("notebook", "image", "file")
        ):
            return None
        try:
            _validate_relative_name(name)
        except ValueError:
            return None
        return cls(
            id=artifact_id,
            name=name,
            media_type=media_type,
            size=size,
            sha256=digest,
            uri=uri,
            kind=kind,
        )


@dataclass(frozen=True, slots=True)
class ArtifactPublication:
    """Durable result of publishing one artifact bundle."""

    id: str
    kind: str
    artifacts: tuple[StoredArtifact, ...]
    created_at: float | None = None

    def __post_init__(self) -> None:
        if _PUBLICATION_ID_RE.fullmatch(self.id) is None:
            raise ValueError("Invalid artifact publication ID")
        if not self.kind.strip():
            raise ValueError("Artifact publication kind cannot be empty")
        parsed = _stored_artifacts_from_values(
            [artifact.to_dict() for artifact in self.artifacts]
        )
        if parsed != self.artifacts:
            raise ValueError("Invalid artifact publication descriptors")
        if self.created_at is not None and (
            isinstance(self.created_at, bool) or self.created_at < 0
        ):
            raise ValueError("Invalid artifact publication creation time")

    def to_metadata(self) -> dict[str, object]:
        return {
            "artifact_publication": {
                "id": self.id,
                "kind": self.kind,
                "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            }
        }


@dataclass(frozen=True, slots=True)
class LoadedArtifact:
    """Verified artifact bytes and their authoritative descriptor."""

    descriptor: StoredArtifact
    data: bytes


def validate_loaded_artifact(
    loaded: LoadedArtifact,
    *,
    expected: StoredArtifact | None = None,
) -> LoadedArtifact:
    """Verify artifact bytes against authoritative and expected descriptors."""

    descriptor = loaded.descriptor
    if StoredArtifact.from_dict(descriptor.to_dict()) is None:
        raise ArtifactUnavailable()
    if len(loaded.data) != descriptor.size:
        raise ArtifactUnavailable()
    if hashlib.sha256(loaded.data).hexdigest() != descriptor.sha256:
        raise ArtifactUnavailable()
    if expected is not None and descriptor != expected:
        raise ArtifactUnavailable()
    return loaded


class ArtifactStoreError(Exception):
    """Base error for artifact persistence."""


class ArtifactPublicationError(ArtifactStoreError):
    """Artifact publication failed."""

    def __init__(self, message: str = "Artifacts could not be published.") -> None:
        super().__init__(message)


class ArtifactUnavailable(ArtifactStoreError):
    """An artifact is missing, unauthorized, malformed, or corrupt."""

    def __init__(self, message: str = "Artifact is unavailable.") -> None:
        super().__init__(message)


@runtime_checkable
class ArtifactStore(Protocol):
    """Application-owned storage boundary for capability artifacts."""

    async def publish(
        self,
        bundle: ArtifactBundle,
        *,
        context: ArtifactContext,
    ) -> ArtifactPublication: ...

    async def get(
        self,
        artifact_id: str,
        *,
        context: ArtifactContext,
    ) -> LoadedArtifact: ...


class InMemoryArtifactStore:
    """Artifact store for tests and short-lived programmatic workflows."""

    def __init__(self) -> None:
        self._artifacts: dict[str, LoadedArtifact] = {}
        self._lock = asyncio.Lock()

    async def publish(
        self,
        bundle: ArtifactBundle,
        *,
        context: ArtifactContext,
    ) -> ArtifactPublication:
        del context
        publication_id = _new_publication_id()
        created_at = time.time()
        artifacts = tuple(
            StoredArtifact(
                id=_new_artifact_id(),
                name=artifact.name,
                media_type=artifact.media_type,
                size=artifact.size,
                sha256=artifact.sha256,
                uri=f"memory://{publication_id}/{artifact.name}",
                kind=artifact.kind,
            )
            for artifact in bundle.artifacts
        )
        async with self._lock:
            self._artifacts.update(
                {
                    descriptor.id: LoadedArtifact(descriptor, bytes(artifact.data))
                    for artifact, descriptor in zip(
                        bundle.artifacts, artifacts, strict=True
                    )
                }
            )
        return ArtifactPublication(
            publication_id,
            bundle.kind,
            artifacts,
            created_at,
        )

    async def get(
        self,
        artifact_id: str,
        *,
        context: ArtifactContext,
    ) -> LoadedArtifact:
        del context
        async with self._lock:
            loaded = self._artifacts.get(artifact_id)
        if loaded is None:
            raise ArtifactUnavailable()
        return validate_loaded_artifact(
            LoadedArtifact(loaded.descriptor, bytes(loaded.data))
        )


class FilesystemArtifactStore:
    """Immutable filesystem artifact storage used by the SQLsaber CLI."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().absolute()

    async def publish(
        self,
        bundle: ArtifactBundle,
        *,
        context: ArtifactContext,
    ) -> ArtifactPublication:
        try:
            return await asyncio.to_thread(self._publish_sync, bundle, context)
        except asyncio.CancelledError:
            raise
        except ArtifactPublicationError:
            raise
        except Exception as exc:
            raise ArtifactPublicationError("Artifacts could not be published.") from exc

    def _publish_sync(
        self,
        bundle: ArtifactBundle,
        context: ArtifactContext,
    ) -> ArtifactPublication:
        publication_id = _new_publication_id()
        self._ensure_root()
        shard = self.root / publication_id[3:5]
        self._ensure_directory(shard)
        target = shard / publication_id
        if target.exists() or target.is_symlink():
            raise ArtifactPublicationError()
        temporary = shard / f".tmp-{publication_id}-{secrets.token_hex(8)}"
        temporary.mkdir(mode=0o700)
        artifact_root = temporary / _FILESYSTEM_ARTIFACTS_DIRECTORY
        stored: list[StoredArtifact] = []
        created_at = time.time()
        try:
            artifact_root.mkdir(mode=0o700)
            for artifact in bundle.artifacts:
                relative = PurePosixPath(artifact.name)
                parent = artifact_root
                for part in relative.parts[:-1]:
                    parent /= part
                    parent.mkdir(mode=0o700, exist_ok=True)
                    parent.chmod(0o700)
                path = artifact_root.joinpath(*relative.parts)
                self._write_durable(path, artifact.data)
                final_path = target.joinpath(
                    _FILESYSTEM_ARTIFACTS_DIRECTORY, *relative.parts
                )
                stored.append(
                    StoredArtifact(
                        id=_new_artifact_id(),
                        name=artifact.name,
                        media_type=artifact.media_type,
                        size=artifact.size,
                        sha256=artifact.sha256,
                        uri=final_path.as_uri(),
                        kind=artifact.kind,
                    )
                )
            manifest = {
                "schema_version": _ARTIFACT_SCHEMA_VERSION,
                "id": publication_id,
                "kind": bundle.kind,
                "created_at": created_at,
                "context": {
                    key: value
                    for key, value in {
                        "run_id": context.run_id,
                        "conversation_id": context.conversation_id,
                        "tool_call_id": context.tool_call_id,
                    }.items()
                    if value is not None
                },
                "metadata": _json_safe(bundle.metadata),
                "artifacts": [artifact.to_dict() for artifact in stored],
            }
            manifest_path = temporary / _FILESYSTEM_MANIFEST_NAME
            self._write_durable(
                manifest_path,
                json.dumps(manifest, sort_keys=True).encode("utf-8"),
            )
            self._fsync_tree_directories(artifact_root)
            self._fsync_directory(temporary)
            os.rename(temporary, target)
            self._fsync_directory(shard)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return ArtifactPublication(
            publication_id,
            bundle.kind,
            tuple(stored),
            created_at,
        )

    async def get(
        self,
        artifact_id: str,
        *,
        context: ArtifactContext,
    ) -> LoadedArtifact:
        del context
        try:
            return await asyncio.to_thread(self._get_sync, artifact_id)
        except ArtifactUnavailable:
            raise
        except Exception as exc:
            raise ArtifactUnavailable() from exc

    def _get_sync(self, artifact_id: str) -> LoadedArtifact:
        if _ARTIFACT_ID_RE.fullmatch(artifact_id) is None:
            raise ArtifactUnavailable()
        self._reject_symlink_components(self.root)
        self._require_directory(self.root)
        for shard in self.root.iterdir():
            if shard.name.startswith("."):
                continue
            if shard.is_symlink():
                raise ArtifactUnavailable()
            if re.fullmatch(r"[a-f0-9]{2}", shard.name) is None:
                continue
            self._require_directory(shard)
            for entry in shard.iterdir():
                if entry.name.startswith("."):
                    continue
                if _PUBLICATION_ID_RE.fullmatch(entry.name) is None:
                    continue
                if entry.is_symlink():
                    continue
                try:
                    self._require_directory(entry)
                    names = {child.name for child in entry.iterdir()}
                except (ArtifactUnavailable, OSError):
                    continue
                if entry.name[3:5] != shard.name or names != {
                    _FILESYSTEM_MANIFEST_NAME,
                    _FILESYSTEM_ARTIFACTS_DIRECTORY,
                }:
                    continue
                artifact_root = entry / _FILESYSTEM_ARTIFACTS_DIRECTORY
                try:
                    self._require_directory(artifact_root)
                except ArtifactUnavailable:
                    continue
                try:
                    manifest = json.loads(
                        self._read_regular_file(entry / _FILESYSTEM_MANIFEST_NAME)
                    )
                except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if not isinstance(manifest, Mapping):
                    continue
                publication_id = manifest.get("id")
                if (
                    manifest.get("schema_version") != _ARTIFACT_SCHEMA_VERSION
                    or publication_id != entry.name
                ):
                    continue
                values = manifest.get("artifacts")
                descriptors = _stored_artifacts_from_values(values)
                if descriptors is None:
                    continue
                for descriptor in descriptors:
                    if descriptor.id != artifact_id:
                        continue
                    relative = PurePosixPath(descriptor.name)
                    parent = artifact_root
                    for part in relative.parts[:-1]:
                        parent /= part
                        self._require_directory(parent)
                    path = artifact_root.joinpath(*relative.parts)
                    if descriptor.uri != path.as_uri():
                        raise ArtifactUnavailable()
                    try:
                        return validate_loaded_artifact(
                            LoadedArtifact(
                                descriptor,
                                self._read_regular_file(path),
                            )
                        )
                    except OSError as exc:
                        raise ArtifactUnavailable() from exc
        raise ArtifactUnavailable()

    async def iter_publications(self) -> list[ArtifactPublication]:
        try:
            return await asyncio.to_thread(self._iter_publications_sync)
        except Exception as exc:
            raise ArtifactUnavailable() from exc

    def _iter_publications_sync(self) -> list[ArtifactPublication]:
        if not self.root.exists():
            return []
        self._reject_symlink_components(self.root)
        self._require_directory(self.root)
        publications: list[ArtifactPublication] = []
        for shard in self.root.iterdir():
            if shard.name.startswith("."):
                continue
            if shard.is_symlink():
                raise ArtifactUnavailable()
            if re.fullmatch(r"[a-f0-9]{2}", shard.name) is None:
                continue
            self._require_directory(shard)
            for entry in shard.iterdir():
                if entry.name.startswith("."):
                    continue
                if _PUBLICATION_ID_RE.fullmatch(entry.name) is None:
                    continue
                if entry.is_symlink():
                    continue
                try:
                    self._require_directory(entry)
                except ArtifactUnavailable:
                    continue
                if entry.name[3:5] != shard.name:
                    continue
                try:
                    manifest = json.loads(
                        self._read_regular_file(entry / _FILESYSTEM_MANIFEST_NAME)
                    )
                except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if not isinstance(manifest, Mapping):
                    continue
                publication_id = manifest.get("id")
                kind = manifest.get("kind")
                created_at = manifest.get("created_at")
                values = manifest.get("artifacts")
                if (
                    manifest.get("schema_version") != _ARTIFACT_SCHEMA_VERSION
                    or publication_id != entry.name
                    or not isinstance(kind, str)
                    or not kind.strip()
                    or not isinstance(created_at, (int, float))
                    or isinstance(created_at, bool)
                    or not isinstance(values, list)
                ):
                    continue
                descriptors = _stored_artifacts_from_values(values)
                if descriptors is not None:
                    publications.append(
                        ArtifactPublication(
                            publication_id,
                            kind,
                            descriptors,
                            float(created_at),
                        )
                    )
        return publications

    async def delete_publication(self, publication_id: str) -> None:
        try:
            await asyncio.to_thread(self._delete_publication_sync, publication_id)
        except FileNotFoundError:
            return
        except ArtifactUnavailable:
            raise
        except Exception as exc:
            raise ArtifactUnavailable() from exc

    def _delete_publication_sync(self, publication_id: str) -> None:
        if _PUBLICATION_ID_RE.fullmatch(publication_id) is None:
            raise ArtifactUnavailable()
        self._reject_symlink_components(self.root)
        if not self.root.exists():
            return
        self._require_directory(self.root)
        shard = self.root / publication_id[3:5]
        if not shard.exists() and not shard.is_symlink():
            return
        self._require_directory(shard)
        entry = shard / publication_id
        if not entry.exists() and not entry.is_symlink():
            return
        self._require_directory(entry)
        tombstone = shard / f".tombstone-{publication_id}-{secrets.token_hex(8)}"
        os.rename(entry, tombstone)
        self._fsync_directory(shard)
        shutil.rmtree(tombstone, ignore_errors=True)

    async def cleanup_stale_workdirs(self, *, older_than: float) -> None:
        try:
            await asyncio.to_thread(self._cleanup_stale_workdirs_sync, older_than)
        except Exception as exc:
            raise ArtifactUnavailable() from exc

    def _cleanup_stale_workdirs_sync(self, older_than: float) -> None:
        if not self.root.exists():
            return
        self._reject_symlink_components(self.root)
        self._require_directory(self.root)
        for shard in self.root.iterdir():
            if shard.name.startswith("."):
                continue
            if shard.is_symlink():
                raise ArtifactUnavailable()
            if re.fullmatch(r"[a-f0-9]{2}", shard.name) is None:
                continue
            self._require_directory(shard)
            for entry in shard.iterdir():
                if not entry.name.startswith((".tmp-", ".tombstone-")):
                    continue
                try:
                    info = entry.lstat()
                except OSError:
                    continue
                if stat.S_ISDIR(info.st_mode) and info.st_mtime < older_than:
                    shutil.rmtree(entry, ignore_errors=True)

    def _ensure_root(self) -> None:
        self._reject_symlink_components(self.root)
        self.root.mkdir(parents=True, mode=0o700, exist_ok=True)
        self._require_directory(self.root)
        try:
            self.root.chmod(0o700)
        except OSError:
            pass

    @staticmethod
    def _reject_symlink_components(path: Path) -> None:
        for component in reversed([path, *path.parents]):
            try:
                mode = component.lstat().st_mode
            except FileNotFoundError:
                continue
            if stat.S_ISLNK(mode):
                raise ArtifactUnavailable()

    @staticmethod
    def _ensure_directory(path: Path) -> None:
        if path.is_symlink():
            raise ArtifactUnavailable()
        path.mkdir(mode=0o700, exist_ok=True)
        FilesystemArtifactStore._require_directory(path)
        try:
            path.chmod(0o700)
        except OSError:
            pass

    @staticmethod
    def _require_directory(path: Path) -> None:
        try:
            mode = path.lstat().st_mode
        except OSError as exc:
            raise ArtifactUnavailable() from exc
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            raise ArtifactUnavailable()

    @staticmethod
    def _write_durable(path: Path, data: bytes) -> None:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "wb", closefd=False) as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            os.close(fd)

    @classmethod
    def _fsync_tree_directories(cls, root: Path) -> None:
        directories = [root, *(path for path in root.rglob("*") if path.is_dir())]
        for directory in sorted(
            directories,
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            cls._fsync_directory(directory)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        if not hasattr(os, "O_DIRECTORY"):
            return
        try:
            fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        except OSError:
            return
        try:
            os.fsync(fd)
        except OSError:
            pass
        finally:
            os.close(fd)

    @staticmethod
    def _read_regular_file(path: Path) -> bytes:
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode):
            raise ArtifactUnavailable()
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(path, flags)
        try:
            opened = os.fstat(fd)
            current = path.lstat()
            if (
                not stat.S_ISREG(opened.st_mode)
                or not stat.S_ISREG(current.st_mode)
                or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
                or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)
            ):
                raise ArtifactUnavailable()
            with os.fdopen(fd, "rb", closefd=False) as stream:
                return stream.read()
        finally:
            os.close(fd)


def artifact_publication_from_metadata(
    metadata: object,
) -> ArtifactPublication | None:
    """Parse one minimal publication reference from tool-return metadata."""

    if not isinstance(metadata, Mapping):
        return None
    mapping = cast(Mapping[str, Any], metadata)
    value = mapping.get("artifact_publication")
    if not isinstance(value, Mapping):
        return None
    publication_id = value.get("id")
    kind = value.get("kind")
    artifacts = value.get("artifacts")
    if (
        not isinstance(publication_id, str)
        or _PUBLICATION_ID_RE.fullmatch(publication_id) is None
        or not isinstance(kind, str)
        or not kind.strip()
        or not isinstance(artifacts, Sequence)
        or isinstance(artifacts, (str, bytes))
    ):
        return None
    parsed = _stored_artifacts_from_values(artifacts)
    if parsed is None:
        return None
    return ArtifactPublication(publication_id, kind, parsed)


def artifacts_from_metadata(metadata: object) -> list[StoredArtifact]:
    """Parse durable artifact references from tool-return metadata."""

    publication = artifact_publication_from_metadata(metadata)
    return list(publication.artifacts) if publication is not None else []


def _stored_artifacts_from_values(
    values: object,
) -> tuple[StoredArtifact, ...] | None:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return None
    parsed: list[StoredArtifact] = []
    ids: set[str] = set()
    names: set[str] = set()
    for value in values:
        artifact = StoredArtifact.from_dict(value)
        if artifact is None or artifact.id in ids or artifact.name in names:
            return None
        ids.add(artifact.id)
        names.add(artifact.name)
        parsed.append(artifact)
    return tuple(parsed)


def _new_publication_id() -> str:
    return f"ap_{secrets.token_hex(16)}"


def _new_artifact_id() -> str:
    return f"ar_{secrets.token_hex(16)}"


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
