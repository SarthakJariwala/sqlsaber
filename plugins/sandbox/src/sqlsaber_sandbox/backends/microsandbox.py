"""Native local microVM backend using the Microsandbox SDK."""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import math
import os
import sys
from collections.abc import Coroutine, Sequence
from typing import Any
import uuid

from ..config import DEFAULT_SANDBOX_IMAGE, SandboxConfig
from .base import CommandResult, SandboxError


class MicrosandboxBackend:
    """One SDK-owned microVM and its native controller exec handle."""

    def __init__(self) -> None:
        self._sdk: Any | None = None
        self._sandbox: Any | None = None
        self._name: str | None = None
        self._config: SandboxConfig | None = None
        self._controller_handle: Any | None = None
        self._controller_drain: asyncio.Task[bool] | None = None
        self._close_lock = asyncio.Lock()

    async def open(self, config: SandboxConfig) -> None:
        if self._name is not None:
            raise SandboxError("Microsandbox is already open")
        if config.provider not in {None, "microsandbox"}:
            raise ValueError(
                f"Microsandbox backend cannot use provider {config.provider!r}"
            )
        if config.gpu is not None:
            raise ValueError("Microsandbox backend does not support GPU allocation")

        cpus: int | None = None
        if config.cpu_cores is not None:
            cpus = _whole_cpus(config.cpu_cores)

        _check_host_support()
        sdk = _load_microsandbox()
        self._sdk = sdk
        self._config = config
        self._name = f"sqlsaber-sandbox-{uuid.uuid4().hex}"
        kwargs: dict[str, Any] = {
            "image": config.image or DEFAULT_SANDBOX_IMAGE,
            "network": sdk.Network.allow_all(),
            "security": sdk.SecurityProfile.RESTRICTED,
            "ephemeral": True,
            "max_duration": (
                float(config.max_lifetime_seconds)
                if config.max_lifetime_seconds is not None
                else None
            ),
        }
        if cpus is not None:
            kwargs["cpus"] = cpus
        if config.memory_mb is not None:
            kwargs["memory"] = config.memory_mb

        try:
            self._sandbox = await sdk.Sandbox.create(self._name, **kwargs)
        except asyncio.CancelledError:
            await _preserve_primary(self.close())
            raise
        except Exception as exc:
            await _preserve_primary(self.close())
            raise SandboxError(f"Microsandbox could not start: {exc}") from exc

    async def execute(self, command: str, *, timeout: float) -> CommandResult:
        sandbox = self._require_sandbox()
        handle: Any | None = None
        try:
            async with asyncio.timeout(timeout):
                handle = await sandbox.exec_stream(
                    "sh", ["-lc", command], timeout=timeout
                )
                stdout, stderr, exit_code = await _collect(handle)
        except asyncio.CancelledError:
            if handle is None or not await _kill_handle(handle):
                await _preserve_primary(self.close())
            raise
        except TimeoutError:
            if handle is None or not await _kill_handle(handle):
                await _preserve_primary(self.close())
            raise
        except Exception as exc:
            if handle is None or not await _kill_handle(handle):
                await _preserve_primary(self.close())
            if _is_sdk_exception(exc, self._sdk, "ExecTimeoutError"):
                raise TimeoutError(
                    f"Microsandbox command timed out after {timeout} seconds"
                ) from exc
            raise SandboxError(f"Microsandbox command transport failed: {exc}") from exc
        return CommandResult(
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
            exit_code=exit_code,
        )

    async def upload(self, data: bytes, path: str) -> None:
        sandbox = self._require_sandbox()
        try:
            await sandbox.fs.write(path, data)
        except Exception as exc:
            raise SandboxError(f"Microsandbox upload failed: {exc}") from exc

    async def download(self, path: str) -> bytes:
        sandbox = self._require_sandbox()
        try:
            data = await sandbox.fs.read(path)
        except Exception as exc:
            raise SandboxError(f"Microsandbox download failed: {exc}") from exc
        if not isinstance(data, bytes):
            raise SandboxError("Microsandbox download returned non-binary data")
        return data

    async def start_controller(self, argv: Sequence[str]) -> None:
        sandbox = self._require_sandbox()
        arguments = tuple(argv)
        if not arguments:
            raise ValueError("Controller argv cannot be empty")
        if self._controller_handle is not None:
            raise SandboxError("Microsandbox controller has already been started")
        try:
            handle = await sandbox.exec_stream(arguments[0], list(arguments[1:]))
        except Exception as exc:
            raise SandboxError(
                f"Microsandbox controller could not start: {exc}"
            ) from exc
        self._controller_handle = handle
        self._controller_drain = asyncio.create_task(_drain_controller(handle))
        await asyncio.sleep(0)

    async def close(self) -> None:
        async with self._close_lock:
            controller_stopped = await self._stop_controller()
            sdk = self._sdk
            name = self._name
            sandbox = self._sandbox
            if sdk is None or name is None:
                if not controller_stopped:
                    raise SandboxError("Microsandbox controller cleanup failed")
                self._clear()
                return

            target = sandbox
            if target is None:
                try:
                    target = await sdk.Sandbox.get(name)
                except Exception as exc:
                    if _is_sdk_exception(exc, sdk, "SandboxNotFoundError"):
                        self._clear()
                        return
                    raise SandboxError(
                        f"Microsandbox cleanup lookup failed: {exc}"
                    ) from exc

            timeout = (
                float(self._config.transport_seconds)
                if self._config is not None
                else 30.0
            )
            try:
                await target.destroy(force=True, timeout=timeout)
            except Exception as exc:
                if _is_sdk_exception(exc, sdk, "SandboxNotFoundError"):
                    self._clear()
                    return
                raise SandboxError(f"Microsandbox cleanup failed: {exc}") from exc
            self._clear()

    def _require_sandbox(self) -> Any:
        if self._sandbox is None:
            raise SandboxError("Microsandbox is not open")
        return self._sandbox

    async def _stop_controller(self) -> bool:
        handle = self._controller_handle
        drain = self._controller_drain
        if handle is None:
            return True
        timeout = min(
            float(self._config.transport_seconds) if self._config is not None else 5.0,
            5.0,
        )
        stopped = False
        if drain is not None and drain.done():
            with contextlib.suppress(BaseException):
                stopped = drain.result()
        if not stopped:
            stopped = await _kill_handle(handle, timeout=timeout)
        if not stopped:
            return False
        if drain is not None:
            if drain.done():
                await asyncio.gather(drain, return_exceptions=True)
            else:
                try:
                    await asyncio.wait_for(asyncio.shield(drain), timeout=timeout)
                except TimeoutError:
                    drain.cancel()
                    await asyncio.gather(drain, return_exceptions=True)
        self._controller_handle = None
        self._controller_drain = None
        return True

    def _clear(self) -> None:
        self._sandbox = None
        self._name = None
        self._sdk = None
        self._config = None
        self._controller_handle = None
        self._controller_drain = None


async def _collect(handle: Any) -> tuple[bytes, bytes, int]:
    stdout = bytearray()
    stderr = bytearray()
    exit_code: int | None = None
    async for event in handle:
        kind = str(event.event_type)
        if kind == "stdout" and event.data:
            stdout.extend(event.data)
        elif kind in {"stderr", "failed", "stdin_error"} and event.data:
            stderr.extend(event.data)
        elif kind == "exited":
            exit_code = event.code
    if exit_code is None:
        exit_code, _ = await handle.wait()
    return bytes(stdout), bytes(stderr), exit_code


async def _drain_controller(handle: Any) -> bool:
    try:
        async for _ in handle:
            pass
        await handle.wait()
        return True
    except Exception:
        return False


async def _kill_handle(handle: Any | None, *, timeout: float = 5.0) -> bool:
    if handle is None:
        return True
    # Native SDK methods return asyncio Futures rather than coroutines.
    task = asyncio.ensure_future(handle.kill())
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
    except BaseException:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return False
    return True


async def _preserve_primary(cleanup: Coroutine[Any, Any, None]) -> None:
    task = asyncio.create_task(cleanup)
    with contextlib.suppress(BaseException):
        await asyncio.shield(task)


def _whole_cpus(value: float) -> int:
    number = float(value)
    if not math.isfinite(number) or number < 1 or not number.is_integer():
        raise ValueError("Microsandbox requires cpu_cores to be a whole number")
    return int(number)


def _check_host_support() -> None:
    if sys.platform == "linux" and not os.access("/dev/kvm", os.R_OK | os.W_OK):
        raise SandboxError(
            "Microsandbox requires readable and writable /dev/kvm on Linux"
        )


def _is_sdk_exception(exc: Exception, sdk: Any | None, name: str) -> bool:
    candidate = getattr(sdk, name, None)
    return isinstance(candidate, type) and isinstance(exc, candidate)


def _load_microsandbox() -> Any:
    try:
        return importlib.import_module("microsandbox")
    except ImportError as exc:
        raise SandboxError(
            "Microsandbox backend requires microsandbox>=0.6.6,<0.7"
        ) from exc
