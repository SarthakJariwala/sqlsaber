"""Thread-aware mark-and-sweep maintenance for CLI artifacts."""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import time
from dataclasses import dataclass
from pathlib import Path

from sqlsaber.artifact_resolution import artifact_references_from_messages
from sqlsaber.artifacts import FilesystemArtifactStore
from sqlsaber.config.logging import get_logger
from sqlsaber.threads.storage import ThreadStorage

logger = get_logger(__name__)
DEFAULT_GRACE_SECONDS = 24 * 60 * 60
SWEEP_INTERVAL_SECONDS = 24 * 60 * 60
STALE_LOCK_SECONDS = 60 * 60


@dataclass(frozen=True, slots=True)
class ArtifactGCResult:
    deleted: int = 0
    skipped: bool = False
    complete: bool = True


async def collect_cli_artifacts(
    thread_storage: ThreadStorage,
    store: FilesystemArtifactStore,
    *,
    force: bool = False,
    grace_seconds: float = DEFAULT_GRACE_SECONDS,
    now: float | None = None,
) -> ArtifactGCResult:
    """Delete old orphans, aborting on any snapshot parse failure."""

    current = time.time() if now is None else now
    lock_token = await asyncio.to_thread(_acquire_lock, store.root, current)
    if lock_token is None:
        return ArtifactGCResult(skipped=True)
    lock_path = store.root / ".maintenance.lock"
    marker = store.root / ".last-successful-sweep"
    try:
        if not force and await asyncio.to_thread(_recent_marker, marker, current):
            return ArtifactGCResult(skipped=True)
        try:
            snapshots = await thread_storage.get_all_thread_messages_strict()
        except Exception as exc:
            logger.warning("artifacts.gc.snapshot_failed", error=str(exc))
            return ArtifactGCResult(complete=False)

        live = {
            reference.publication_id
            for messages in snapshots
            for reference in artifact_references_from_messages(messages)
        }
        deleted = 0
        cutoff = current - grace_seconds
        for publication in await store.iter_publications():
            if (
                publication.id in live
                or publication.created_at is None
                or publication.created_at >= cutoff
            ):
                continue
            await store.delete_publication(publication.id)
            deleted += 1
        await store.cleanup_stale_workdirs(older_than=cutoff)
        await asyncio.to_thread(_write_marker, marker, current)
        return ArtifactGCResult(deleted=deleted)
    except Exception as exc:
        logger.warning("artifacts.gc.failed", error=str(exc))
        return ArtifactGCResult(complete=False)
    finally:
        await asyncio.to_thread(_release_lock, lock_path, lock_token)


def _acquire_lock(root: Path, now: float) -> str | None:
    for component in [root, *root.parents]:
        if component.is_symlink():
            return None
    if root.exists() and not root.is_dir():
        return None
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock = root / ".maintenance.lock"
    token = secrets.token_hex(16)
    payload = json.dumps(
        {"pid": os.getpid(), "created_at": now, "token": token}
    ).encode()
    for attempt in range(2):
        try:
            fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            if attempt or not _lock_is_stale(lock, now):
                return None
            try:
                lock.unlink()
            except OSError:
                return None
            continue
        try:
            os.write(fd, payload)
            os.fsync(fd)
        finally:
            os.close(fd)
        return token
    return None


def _lock_is_stale(path: Path, now: float) -> bool:
    try:
        if path.is_symlink() or not path.is_file():
            return False
        value = json.loads(path.read_text(encoding="utf-8"))
        created = value.get("created_at") if isinstance(value, dict) else None
        return isinstance(created, (int, float)) and now - created > STALE_LOCK_SECONDS
    except (OSError, json.JSONDecodeError):
        return False


def _release_lock(path: Path, token: str) -> None:
    try:
        if path.is_symlink() or not path.is_file():
            return
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("token") != token:
            return
        path.unlink()
    except (OSError, json.JSONDecodeError):
        pass


def _recent_marker(path: Path, now: float) -> bool:
    try:
        if path.is_symlink() or not path.is_file():
            return False
        timestamp = float(path.read_text(encoding="ascii"))
    except (OSError, ValueError):
        return False
    return now - timestamp < SWEEP_INTERVAL_SECONDS


def _write_marker(path: Path, now: float) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, f"{now:.6f}".encode("ascii"))
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(temporary, path)
