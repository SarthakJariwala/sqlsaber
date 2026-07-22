"""Durable, immutable storage for complete SQL query results."""

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
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from sqlsaber.utils.json_utils import json_dumps

QueryResultId = str
QUERY_RESULT_MEDIA_TYPE = "application/vnd.sqlsaber.query-result+json"
QUERY_RESULT_SCHEMA_VERSION = 1
MAX_MODEL_QUERY_RESULT_BYTES = 12 * 1024
MAX_CANONICAL_QUERY_RESULT_BYTES = 100 * 1024 * 1024
_RESULT_ID_RE = re.compile(r"^qr_[a-f0-9]{32}$")
_RESULT_FILE_RE = re.compile(r"^result_[A-Za-z0-9._-]{1,180}\.json$")


class QueryResultUnavailable(Exception):
    """A query result is missing, unauthorized, malformed, or corrupt.

    The intentionally uniform public message does not disclose whether an ID exists.
    """

    def __init__(self, message: str = "Query result is unavailable.") -> None:
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class QueryResultContext:
    run_id: str | None = None
    conversation_id: str | None = None
    tool_call_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class QueryResultData:
    data: bytes
    media_type: str = QUERY_RESULT_MEDIA_TYPE


@dataclass(frozen=True, slots=True)
class StoredQueryResult:
    id: str
    file: str
    media_type: str
    size: int
    sha256: str
    row_count: int
    columns: tuple[str, ...]
    database_name: str | None = None
    created_at: float | None = None

    def to_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "id": self.id,
            "file": self.file,
            "media_type": self.media_type,
            "size": self.size,
            "sha256": self.sha256,
            "row_count": self.row_count,
            "columns": list(self.columns),
        }
        if self.database_name is not None:
            value["database_name"] = self.database_name
        if self.created_at is not None:
            value["created_at"] = self.created_at
        return value

    @classmethod
    def from_dict(cls, value: object) -> StoredQueryResult | None:
        if not isinstance(value, dict):
            return None
        raw: dict[str, object] = {str(key): item for key, item in value.items()}
        try:
            result_id = raw["id"]
            file = raw["file"]
            media_type = raw["media_type"]
            size = raw["size"]
            digest = raw["sha256"]
            row_count = raw["row_count"]
            columns = raw["columns"]
        except KeyError:
            return None
        if not isinstance(result_id, str) or not valid_query_result_id(result_id):
            return None
        if not isinstance(file, str) or not valid_query_result_file(file):
            return None
        if not isinstance(media_type, str) or media_type != QUERY_RESULT_MEDIA_TYPE:
            return None
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            return None
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            return None
        if (
            not isinstance(row_count, int)
            or isinstance(row_count, bool)
            or row_count < 0
        ):
            return None
        if not isinstance(columns, (list, tuple)) or any(
            not isinstance(column, str) for column in columns
        ):
            return None
        database_name = raw.get("database_name")
        if database_name is not None and not isinstance(database_name, str):
            return None
        created_at = raw.get("created_at")
        if created_at is not None and (
            not isinstance(created_at, (int, float)) or isinstance(created_at, bool)
        ):
            return None
        return cls(
            id=result_id,
            file=file,
            media_type=media_type,
            size=size,
            sha256=digest,
            row_count=row_count,
            columns=tuple(str(column) for column in columns),
            database_name=database_name,
            created_at=float(created_at) if created_at is not None else None,
        )


@dataclass(frozen=True, slots=True)
class LoadedQueryResult:
    descriptor: StoredQueryResult
    data: bytes

    def payload(self) -> dict[str, Any]:
        try:
            value = json.loads(self.data)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise QueryResultUnavailable() from exc
        if not isinstance(value, dict):
            raise QueryResultUnavailable()
        return value

    def rows(self) -> list[dict[str, Any]]:
        value = self.payload().get("results")
        if not isinstance(value, list):
            raise QueryResultUnavailable()
        rows: list[dict[str, Any]] = []
        for row in value:
            if isinstance(row, dict):
                rows.append({str(key): item for key, item in row.items()})
            else:
                rows.append({"value": row})
        return rows


@runtime_checkable
class QueryResultStore(Protocol):
    async def put(
        self,
        result: QueryResultData,
        *,
        descriptor: StoredQueryResult,
        context: QueryResultContext,
    ) -> StoredQueryResult: ...

    async def get(
        self,
        result_id: str,
        *,
        context: QueryResultContext,
    ) -> LoadedQueryResult: ...


def new_query_result_id() -> str:
    return f"qr_{secrets.token_hex(16)}"


def valid_query_result_id(value: str) -> bool:
    return _RESULT_ID_RE.fullmatch(value) is not None


def valid_query_result_file(value: str) -> bool:
    return (
        _RESULT_FILE_RE.fullmatch(value) is not None
        and Path(value).name == value
        and value not in {".", ".."}
    )


def logical_result_file(tool_call_id: str | None, result_id: str) -> str:
    if tool_call_id and re.fullmatch(r"[A-Za-z0-9._-]{1,180}", tool_call_id):
        return f"result_{tool_call_id}.json"
    return f"result_{result_id}.json"


def query_result_columns(rows: list[object]) -> tuple[str, ...]:
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        keys = row.keys() if isinstance(row, Mapping) else ("value",)
        for key in keys:
            name = str(key)
            if name not in seen:
                seen.add(name)
                columns.append(name)
    return tuple(columns)


def descriptor_for_data(
    data: bytes,
    *,
    result_id: str,
    file: str,
    row_count: int,
    columns: tuple[str, ...],
    database_name: str | None = None,
    created_at: float | None = None,
    media_type: str = QUERY_RESULT_MEDIA_TYPE,
) -> StoredQueryResult:
    return StoredQueryResult(
        id=result_id,
        file=file,
        media_type=media_type,
        size=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        row_count=row_count,
        columns=columns,
        database_name=database_name,
        created_at=created_at,
    )


def _validated_authoritative_descriptor(
    result: QueryResultData, descriptor: StoredQueryResult
) -> StoredQueryResult:
    if StoredQueryResult.from_dict(descriptor.to_dict()) is None:
        raise QueryResultUnavailable()
    if not valid_query_result_id(descriptor.id):
        raise QueryResultUnavailable()
    if not valid_query_result_file(descriptor.file):
        raise QueryResultUnavailable()
    if result.media_type != QUERY_RESULT_MEDIA_TYPE:
        raise QueryResultUnavailable()
    if len(result.data) > MAX_CANONICAL_QUERY_RESULT_BYTES:
        raise QueryResultUnavailable("Query result exceeds the storage size limit.")
    return replace(
        descriptor,
        media_type=result.media_type,
        size=len(result.data),
        sha256=hashlib.sha256(result.data).hexdigest(),
        created_at=(
            descriptor.created_at if descriptor.created_at is not None else time.time()
        ),
    )


def validate_loaded_query_result(
    loaded: LoadedQueryResult,
    *,
    expected: StoredQueryResult | None = None,
) -> LoadedQueryResult:
    descriptor = loaded.descriptor
    if StoredQueryResult.from_dict(descriptor.to_dict()) is None:
        raise QueryResultUnavailable()
    if len(loaded.data) != descriptor.size:
        raise QueryResultUnavailable()
    if hashlib.sha256(loaded.data).hexdigest() != descriptor.sha256:
        raise QueryResultUnavailable()
    if expected is not None and (
        descriptor.id != expected.id
        or descriptor.file != expected.file
        or descriptor.media_type != expected.media_type
        or descriptor.size != expected.size
        or descriptor.sha256 != expected.sha256
    ):
        raise QueryResultUnavailable()
    return loaded


class InMemoryQueryResultStore:
    """Session-local immutable query result storage."""

    def __init__(self) -> None:
        self._results: dict[str, LoadedQueryResult] = {}
        self._lock = asyncio.Lock()

    async def put(
        self,
        result: QueryResultData,
        *,
        descriptor: StoredQueryResult,
        context: QueryResultContext,
    ) -> StoredQueryResult:
        del context
        authoritative = _validated_authoritative_descriptor(result, descriptor)
        async with self._lock:
            if authoritative.id in self._results:
                raise QueryResultUnavailable()
            self._results[authoritative.id] = LoadedQueryResult(
                authoritative, bytes(result.data)
            )
        return authoritative

    async def get(
        self,
        result_id: str,
        *,
        context: QueryResultContext,
    ) -> LoadedQueryResult:
        del context
        if not valid_query_result_id(result_id):
            raise QueryResultUnavailable()
        async with self._lock:
            loaded = self._results.get(result_id)
        if loaded is None:
            raise QueryResultUnavailable()
        return validate_loaded_query_result(
            LoadedQueryResult(loaded.descriptor, bytes(loaded.data))
        )


class FilesystemQueryResultStore:
    """Crash-aware immutable filesystem storage used by the SQLsaber CLI."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser().absolute()

    def _entry_path(self, result_id: str) -> Path:
        if not valid_query_result_id(result_id):
            raise QueryResultUnavailable()
        return self.root / result_id[3:5] / result_id

    async def put(
        self,
        result: QueryResultData,
        *,
        descriptor: StoredQueryResult,
        context: QueryResultContext,
    ) -> StoredQueryResult:
        authoritative = _validated_authoritative_descriptor(result, descriptor)
        try:
            return await asyncio.to_thread(
                self._put_sync, result.data, authoritative, context
            )
        except asyncio.CancelledError:
            raise
        except QueryResultUnavailable:
            raise
        except Exception as exc:
            raise QueryResultUnavailable() from exc

    def _put_sync(
        self,
        data: bytes,
        descriptor: StoredQueryResult,
        context: QueryResultContext,
    ) -> StoredQueryResult:
        self._ensure_root()
        shard = self.root / descriptor.id[3:5]
        self._ensure_directory(shard)
        final = shard / descriptor.id
        if final.exists() or final.is_symlink():
            raise QueryResultUnavailable()
        temporary = shard / f".tmp-{descriptor.id}-{secrets.token_hex(8)}"
        try:
            temporary.mkdir(mode=0o700)
            result_path = temporary / "result.json"
            manifest_path = temporary / "manifest.json"
            self._write_durable(result_path, data)
            manifest = {
                "schema_version": QUERY_RESULT_SCHEMA_VERSION,
                "descriptor": descriptor.to_dict(),
                "context": {
                    key: value
                    for key, value in {
                        "run_id": context.run_id,
                        "conversation_id": context.conversation_id,
                        "tool_call_id": context.tool_call_id,
                    }.items()
                    if value is not None
                },
            }
            self._write_durable(
                manifest_path,
                json_dumps(manifest, ensure_ascii=False, sort_keys=True).encode(
                    "utf-8"
                ),
            )
            self._fsync_directory(temporary)
            os.rename(temporary, final)
            self._fsync_directory(shard)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return descriptor

    async def get(
        self,
        result_id: str,
        *,
        context: QueryResultContext,
    ) -> LoadedQueryResult:
        del context
        try:
            return await asyncio.to_thread(self._get_sync, result_id)
        except QueryResultUnavailable:
            raise
        except Exception as exc:
            raise QueryResultUnavailable() from exc

    def _get_sync(self, result_id: str) -> LoadedQueryResult:
        self._validate_root()
        entry = self._entry_path(result_id)
        self._require_directory(entry.parent)
        self._require_directory(entry)
        manifest_path = entry / "manifest.json"
        result_path = entry / "result.json"
        try:
            names = {child.name for child in entry.iterdir()}
        except OSError as exc:
            raise QueryResultUnavailable() from exc
        if names != {"manifest.json", "result.json"}:
            raise QueryResultUnavailable()
        try:
            manifest = json.loads(self._read_regular_file(manifest_path))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise QueryResultUnavailable() from exc
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema_version") != QUERY_RESULT_SCHEMA_VERSION
        ):
            raise QueryResultUnavailable()
        descriptor = StoredQueryResult.from_dict(manifest.get("descriptor"))
        if descriptor is None or descriptor.id != result_id:
            raise QueryResultUnavailable()
        try:
            data = self._read_regular_file(result_path)
        except OSError as exc:
            raise QueryResultUnavailable() from exc
        return validate_loaded_query_result(LoadedQueryResult(descriptor, data))

    async def iter_descriptors(self) -> list[StoredQueryResult]:
        try:
            return await asyncio.to_thread(self._iter_descriptors_sync)
        except Exception as exc:
            raise QueryResultUnavailable() from exc

    def _iter_descriptors_sync(self) -> list[StoredQueryResult]:
        if not self.root.exists():
            return []
        self._validate_root()
        descriptors: list[StoredQueryResult] = []
        for shard in self.root.iterdir():
            if shard.name.startswith("."):
                continue
            try:
                self._require_directory(shard)
            except QueryResultUnavailable:
                continue
            for entry in shard.iterdir():
                if entry.name.startswith((".tmp-", ".tombstone-")):
                    continue
                if not valid_query_result_id(entry.name):
                    continue
                try:
                    descriptors.append(self._get_sync(entry.name).descriptor)
                except QueryResultUnavailable:
                    continue
        return descriptors

    async def delete(self, result_id: str) -> None:
        try:
            await asyncio.to_thread(self._delete_sync, result_id)
        except FileNotFoundError:
            return
        except QueryResultUnavailable:
            raise
        except Exception as exc:
            raise QueryResultUnavailable() from exc

    async def cleanup_stale_workdirs(self, *, older_than: float) -> None:
        try:
            await asyncio.to_thread(self._cleanup_stale_workdirs_sync, older_than)
        except Exception as exc:
            raise QueryResultUnavailable() from exc

    def _cleanup_stale_workdirs_sync(self, older_than: float) -> None:
        if not self.root.exists():
            return
        self._validate_root()
        for shard in self.root.iterdir():
            try:
                self._require_directory(shard)
            except QueryResultUnavailable:
                continue
            for entry in shard.iterdir():
                if not entry.name.startswith((".tmp-", ".tombstone-")):
                    continue
                try:
                    info = entry.lstat()
                except OSError:
                    continue
                if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                    continue
                if info.st_mtime < older_than:
                    shutil.rmtree(entry, ignore_errors=True)

    def _delete_sync(self, result_id: str) -> None:
        self._validate_root()
        entry = self._entry_path(result_id)
        self._require_directory(entry.parent)
        if not entry.exists():
            return
        self._require_directory(entry)
        tombstone = entry.parent / f".tombstone-{result_id}-{secrets.token_hex(8)}"
        os.rename(entry, tombstone)
        self._fsync_directory(entry.parent)
        shutil.rmtree(tombstone, ignore_errors=True)

    def _ensure_root(self) -> None:
        self._reject_symlink_components(self.root)
        self.root.mkdir(parents=True, mode=0o700, exist_ok=True)
        self._require_directory(self.root)
        try:
            self.root.chmod(0o700)
        except OSError:
            pass

    def _validate_root(self) -> None:
        self._reject_symlink_components(self.root)
        self._require_directory(self.root)

    @staticmethod
    def _reject_symlink_components(path: Path) -> None:
        components = [path, *path.parents]
        for component in reversed(components):
            try:
                mode = component.lstat().st_mode
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise QueryResultUnavailable() from exc
            if stat.S_ISLNK(mode):
                raise QueryResultUnavailable()

    @staticmethod
    def _ensure_directory(path: Path) -> None:
        if path.is_symlink():
            raise QueryResultUnavailable()
        path.mkdir(mode=0o700, exist_ok=True)
        FilesystemQueryResultStore._require_directory(path)
        try:
            path.chmod(0o700)
        except OSError:
            pass

    @staticmethod
    def _require_directory(path: Path) -> None:
        try:
            mode = path.lstat().st_mode
        except OSError as exc:
            raise QueryResultUnavailable() from exc
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            raise QueryResultUnavailable()

    @staticmethod
    def _read_regular_file(path: Path) -> bytes:
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(path, flags)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise QueryResultUnavailable()
            with os.fdopen(fd, "rb", closefd=False) as stream:
                return stream.read()
        finally:
            os.close(fd)

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


def _projection_json(value: Mapping[str, Any]) -> bytes:
    return json_dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def build_model_projection(
    canonical_payload: Mapping[str, Any],
    descriptor: StoredQueryResult,
    *,
    max_bytes: int = MAX_MODEL_QUERY_RESULT_BYTES,
) -> dict[str, Any]:
    """Build a stable, valid JSON projection within a hard UTF-8 byte budget."""

    if max_bytes < 256:
        raise ValueError("max_bytes must be at least 256")
    rows_value = canonical_payload.get("results")
    rows = rows_value if isinstance(rows_value, list) else []
    base: dict[str, Any] = {
        "success": True,
        "result_id": descriptor.id,
        "file": descriptor.file,
        "returned_rows": descriptor.row_count,
        "columns": list(descriptor.columns),
    }
    if canonical_payload.get("auto_limit_applied") is True:
        base["auto_limit_applied"] = True

    complete = {**base, "results": rows, "results_truncated": False}
    if len(_projection_json(complete)) <= max_bytes:
        return complete

    warning = (
        "Preview only; omitted rows must not be used for whole-dataset statistics. "
        "Use the result handle with an analysis tool."
    )
    projected = {
        **base,
        "preview_rows": [],
        "results_truncated": True,
        "warning": warning,
    }
    if len(_projection_json(projected)) > max_bytes:
        projected["columns"] = []
        projected["columns_truncated"] = True
    if len(_projection_json(projected)) > max_bytes:
        projected.pop("warning", None)
        projected["warning"] = (
            "Preview omitted; use the result handle for complete data."
        )
    if len(_projection_json(projected)) > max_bytes:
        minimal: dict[str, Any] = {
            "success": True,
            "result_id": descriptor.id,
            "file": descriptor.file,
            "returned_rows": descriptor.row_count,
            "columns": [],
            "columns_truncated": True,
            "preview_rows": [],
            "results_truncated": True,
        }
        if len(_projection_json(minimal)) > max_bytes:
            raise ValueError("max_bytes is too small for a query result descriptor")
        projected = minimal

    preview: list[Any] = []
    for row in rows:
        candidate = {**projected, "preview_rows": [*preview, row]}
        if len(_projection_json(candidate)) > max_bytes:
            break
        preview.append(row)
    projected["preview_rows"] = preview
    if not preview and rows:
        projected["preview_row_omitted"] = True
        if len(_projection_json(projected)) > max_bytes:
            projected.pop("preview_row_omitted")
    if len(_projection_json(projected)) > max_bytes:  # defensive hard bound
        raise ValueError("Unable to build a bounded query result projection")
    return projected
