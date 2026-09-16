"""Native Modal transport for one persistent SQLsaber sandbox."""

from __future__ import annotations

import asyncio
import contextlib
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
import math
import re
from collections.abc import Sequence
from typing import Any

from ..config import SandboxConfig
from .base import CommandResult, SandboxError

_APP_NAME = "sqlsaber-sandbox"
_PYTHON_VERSION = "3.12"
_MODAL_PLATFORM_TIMEOUT_SECONDS = 24 * 60 * 60


class ModalBackend:
    """Own a Modal Sandbox and its long-running controller process."""

    def __init__(self) -> None:
        self._sandbox: Any | None = None
        self._controller_process: Any | None = None
        self._controller_tasks: tuple[asyncio.Task[Any], ...] = ()
        self._detached_drains: set[asyncio.Future[Any]] = set()

    async def open(self, config: SandboxConfig) -> None:
        if self._sandbox is not None:
            raise SandboxError("Modal sandbox is already open")

        lifetime = config.max_lifetime_seconds
        if lifetime is not None and lifetime > _MODAL_PLATFORM_TIMEOUT_SECONDS:
            raise ValueError("Modal max_lifetime_seconds cannot exceed 86400 seconds")

        modal = _load_modal()
        try:
            app = await modal.App.lookup.aio(_APP_NAME, create_if_missing=True)
            image = (
                modal.Image.from_registry(config.image)
                if config.image is not None
                else modal.Image.debian_slim(python_version=_PYTHON_VERSION)
            )
            options: dict[str, Any] = {
                "app": app,
                "image": image,
                "timeout": lifetime or _MODAL_PLATFORM_TIMEOUT_SECONDS,
                "cpu": config.cpu_cores,
                "memory": config.memory_mb,
                "gpu": config.gpu,
            }
            # The session owns idle expiry, including time spent awaiting its
            # analyst. Provider inactivity is not equivalent to session idle.
            self._sandbox = await modal.Sandbox.create.aio(
                "sleep",
                "infinity",
                **options,
            )
        except Exception as exc:
            raise SandboxError("Could not open Modal sandbox") from exc

    async def execute(self, command: str, *, timeout: float) -> CommandResult:
        sandbox = self._require_sandbox()
        _validate_timeout(timeout)
        drain: asyncio.Future[tuple[Any, Any, Any]] | None = None
        try:
            async with asyncio.timeout(timeout):
                process = await sandbox.exec.aio(
                    "sh",
                    "-lc",
                    command,
                    timeout=math.ceil(timeout),
                )
                drain = asyncio.gather(
                    process.stdout.read.aio(),
                    process.stderr.read.aio(),
                    process.wait.aio(),
                )
                stdout, stderr, exit_code = await asyncio.shield(drain)
        except asyncio.CancelledError:
            if drain is not None:
                self._retain_drain(drain)
            raise
        except TimeoutError as exc:
            if drain is not None:
                self._retain_drain(drain)
            raise SandboxError("Modal command timed out") from exc
        except Exception as exc:
            if drain is not None and not drain.done():
                self._retain_drain(drain)
            raise SandboxError("Modal command failed") from exc

        if not isinstance(exit_code, int):
            raise SandboxError("Modal command returned an invalid result")
        return CommandResult(
            stdout=_as_text(stdout),
            stderr=_as_text(stderr),
            exit_code=exit_code,
        )

    async def upload(self, data: bytes, path: str) -> None:
        sandbox = self._require_sandbox()
        try:
            await sandbox.filesystem.write_bytes.aio(data, path)
        except Exception as exc:
            raise SandboxError("Modal file upload failed") from exc

    async def download(self, path: str) -> bytes:
        sandbox = self._require_sandbox()
        try:
            data = await sandbox.filesystem.read_bytes.aio(path)
        except Exception as exc:
            raise SandboxError("Modal file download failed") from exc
        if not isinstance(data, bytes):
            raise SandboxError("Modal file download returned invalid data")
        return data

    async def start_controller(self, argv: Sequence[str]) -> None:
        sandbox = self._require_sandbox()
        if self._controller_process is not None:
            raise SandboxError("Modal controller is already running")
        if not argv:
            raise ValueError("Controller argv cannot be empty")

        try:
            process = await sandbox.exec.aio(*argv)
            tasks = (
                asyncio.create_task(process.stdout.read.aio()),
                asyncio.create_task(process.stderr.read.aio()),
                asyncio.create_task(process.wait.aio()),
            )
        except Exception as exc:
            raise SandboxError("Could not start Modal controller") from exc

        self._controller_process = process
        self._controller_tasks = tasks

    async def close(self) -> None:
        sandbox = self._sandbox
        if sandbox is None:
            return

        try:
            await sandbox.terminate.aio(wait=True)
            if self._controller_tasks:
                await asyncio.gather(
                    *self._controller_tasks,
                    return_exceptions=True,
                )
            if self._detached_drains:
                await asyncio.gather(
                    *tuple(self._detached_drains),
                    return_exceptions=True,
                )
            await sandbox.detach.aio()
        except Exception as exc:
            raise SandboxError("Could not close Modal sandbox") from exc

        self._sandbox = None
        self._controller_process = None
        self._controller_tasks = ()
        self._detached_drains.clear()

    def _require_sandbox(self) -> Any:
        if self._sandbox is None:
            raise SandboxError("Modal sandbox is not open")
        return self._sandbox

    def _retain_drain(self, drain: asyncio.Future[Any]) -> None:
        self._detached_drains.add(drain)

        def finished(completed: asyncio.Future[Any]) -> None:
            self._detached_drains.discard(completed)
            with contextlib.suppress(BaseException):
                completed.result()

        drain.add_done_callback(finished)


def _load_modal() -> Any:
    try:
        installed = version("modal")
        modal = import_module("modal")
    except (ImportError, PackageNotFoundError) as exc:
        raise SandboxError("Install sqlsaber-sandbox[modal] to use Modal") from exc
    if not _modal_version_supported(installed):
        raise SandboxError("Installed Modal SDK is incompatible")
    return modal


def _modal_version_supported(value: str) -> bool:
    match = re.match(r"^(\d+)\.(\d+)", value)
    return bool(match and int(match[1]) == 1 and int(match[2]) >= 4)


def _validate_timeout(timeout: float) -> None:
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError("timeout must be a finite positive number")


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    raise SandboxError("Modal command returned an invalid result")
