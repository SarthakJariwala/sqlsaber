"""Native Daytona transport compatible with the 0.143 SDK surface."""

from __future__ import annotations

import asyncio
import contextlib
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
import math
import re
import shlex
import uuid
from collections.abc import Sequence
from typing import Any

from ..config import DEFAULT_SANDBOX_IMAGE, SandboxConfig
from .base import CommandResult, SandboxError

_DELETE_TIMEOUT_SECONDS = 60.0
_DELETE_POLL_SECONDS = 0.25


class DaytonaBackend:
    """Own a Daytona client, Sandbox, and retained controller session."""

    def __init__(
        self, *, api_key: str | None = None, api_url: str | None = None
    ) -> None:
        self._api_key = api_key
        self._api_url = api_url
        self._sdk: Any | None = None
        self._client: Any | None = None
        self._sandbox: Any | None = None
        self._sandbox_name: str | None = None
        self._controller_session_id: str | None = None
        self._controller_command_id: str | None = None
        self._sessions: set[str] = set()
        self._transport_seconds = 30.0

    async def open(self, config: SandboxConfig) -> None:
        if self._client is not None or self._sandbox is not None:
            raise SandboxError("Daytona sandbox is already open")

        cpu = _daytona_cpu(config.cpu_cores)
        memory = _daytona_memory_gib(config.memory_mb)
        gpu = _daytona_gpu(config.gpu)

        sdk = _load_daytona()
        self._sdk = sdk
        self._sandbox_name = f"sqlsaber-sandbox-{uuid.uuid4().hex}"
        self._transport_seconds = float(config.transport_seconds)
        try:
            client_options = {
                name: value
                for name, value in (
                    ("api_key", self._api_key),
                    ("api_url", self._api_url),
                )
                if value is not None
            }
            self._client = (
                sdk.AsyncDaytona(sdk.DaytonaConfig(**client_options))
                if client_options
                else sdk.AsyncDaytona()
            )
            params = sdk.CreateSandboxFromImageParams(
                image=config.image or DEFAULT_SANDBOX_IMAGE,
                language="python",
                name=self._sandbox_name,
                labels={"application": "sqlsaber", "purpose": "sandbox-analysis"},
                resources=sdk.Resources(cpu=cpu, memory=memory, gpu=gpu),
                auto_stop_interval=0,
                ephemeral=True,
            )
            self._sandbox = await self._client.create(
                params,
                timeout=config.open_seconds,
            )
        except asyncio.CancelledError:
            await self._recover_partial_sandbox()
            await self._best_effort_close()
            raise
        except Exception as exc:
            await self._recover_partial_sandbox()
            await self._best_effort_close()
            raise SandboxError("Could not open Daytona sandbox") from exc

    async def execute(self, command: str, *, timeout: float) -> CommandResult:
        sandbox = self._require_sandbox()
        sdk = self._require_sdk()
        _validate_timeout(timeout)
        session_id = f"sqlsaber-command-{uuid.uuid4().hex}"
        self._sessions.add(session_id)
        response: Any | None = None
        try:
            async with asyncio.timeout(timeout):
                await sandbox.process.create_session(session_id)
                request = sdk.SessionExecuteRequest(
                    command=command,
                    run_async=False,
                )
                response = await sandbox.process.execute_session_command(
                    session_id,
                    request,
                    timeout=math.ceil(timeout),
                )
        except TimeoutError as exc:
            raise SandboxError("Daytona command timed out") from exc
        except Exception as exc:
            raise SandboxError("Daytona command failed") from exc
        finally:
            await self._delete_session_best_effort(session_id)

        exit_code = response.exit_code
        stdout = response.stdout
        stderr = response.stderr
        if (
            not isinstance(exit_code, int)
            or stdout is not None
            and not isinstance(stdout, str)
            or stderr is not None
            and not isinstance(stderr, str)
        ):
            raise SandboxError("Daytona command returned an invalid result")
        return CommandResult(
            stdout=stdout or "",
            stderr=stderr or "",
            exit_code=exit_code,
        )

    async def upload(self, data: bytes, path: str) -> None:
        sandbox = self._require_sandbox()
        try:
            await sandbox.fs.upload_file(
                data,
                path,
                timeout=math.ceil(self._transport_seconds),
            )
        except Exception as exc:
            raise SandboxError("Daytona file upload failed") from exc

    async def download(self, path: str) -> bytes:
        sandbox = self._require_sandbox()
        try:
            data = await sandbox.fs.download_file(
                path,
                math.ceil(self._transport_seconds),
            )
        except Exception as exc:
            raise SandboxError("Daytona file download failed") from exc
        if not isinstance(data, bytes):
            raise SandboxError("Daytona file download returned invalid data")
        return data

    async def start_controller(self, argv: Sequence[str]) -> None:
        sandbox = self._require_sandbox()
        sdk = self._require_sdk()
        if self._controller_session_id is not None:
            raise SandboxError("Daytona controller is already running")
        if not argv:
            raise ValueError("Controller argv cannot be empty")

        session_id = f"sqlsaber-controller-{uuid.uuid4().hex}"
        self._controller_session_id = session_id
        self._sessions.add(session_id)
        try:
            async with asyncio.timeout(self._transport_seconds):
                await sandbox.process.create_session(session_id)
                response = await sandbox.process.execute_session_command(
                    session_id,
                    sdk.SessionExecuteRequest(
                        command=shlex.join(argv),
                        run_async=True,
                    ),
                    timeout=math.ceil(self._transport_seconds),
                )
            if not isinstance(response.cmd_id, str) or not response.cmd_id:
                raise TypeError("invalid command ID")
        except TimeoutError as exc:
            await self._delete_session_best_effort(session_id)
            raise SandboxError("Daytona controller start timed out") from exc
        except Exception as exc:
            await self._delete_session_best_effort(session_id)
            raise SandboxError("Could not start Daytona controller") from exc

        self._controller_command_id = response.cmd_id

    async def close(self) -> None:
        sandbox = self._sandbox
        client = self._client

        if sandbox is not None:
            try:
                try:
                    await sandbox.delete(timeout=_DELETE_TIMEOUT_SECONDS)
                except Exception as exc:
                    if not self._is_not_found(exc):
                        raise
                else:
                    await self._wait_until_deleted()
            except Exception as exc:
                raise SandboxError("Could not close Daytona sandbox") from exc

            self._sandbox = None
            self._controller_session_id = None
            self._controller_command_id = None
            self._sessions.clear()

        if client is not None:
            try:
                await client.close()
            except Exception as exc:
                raise SandboxError("Could not close Daytona client") from exc
            self._client = None
            self._sdk = None
            self._sandbox_name = None

    async def _wait_until_deleted(self) -> None:
        client = self._client
        name = self._sandbox_name
        if client is None or name is None:
            raise RuntimeError("Daytona cleanup state is incomplete")

        loop = asyncio.get_running_loop()
        deadline = loop.time() + _DELETE_TIMEOUT_SECONDS
        while True:
            try:
                await client.get(name)
            except Exception as exc:
                if self._is_not_found(exc):
                    return
                raise
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError("Daytona sandbox deletion was not confirmed")
            await asyncio.sleep(min(_DELETE_POLL_SECONDS, remaining))

    async def _recover_partial_sandbox(self) -> None:
        if (
            self._sandbox is not None
            or self._client is None
            or self._sandbox_name is None
        ):
            return
        for attempt in range(3):
            try:
                self._sandbox = await self._client.get(self._sandbox_name)
                return
            except Exception as exc:
                if not self._is_not_found(exc):
                    return
            if attempt < 2:
                await asyncio.sleep(0.1)

    async def _best_effort_close(self) -> None:
        with contextlib.suppress(Exception):
            await self.close()

    async def _delete_session_best_effort(self, session_id: str) -> None:
        sandbox = self._sandbox
        if sandbox is None:
            return
        try:
            async with asyncio.timeout(self._transport_seconds):
                await sandbox.process.delete_session(session_id)
        except Exception:
            return
        self._sessions.discard(session_id)
        if self._controller_session_id == session_id:
            self._controller_session_id = None
            self._controller_command_id = None

    def _is_not_found(self, exc: Exception) -> bool:
        sdk = self._sdk
        not_found = getattr(sdk, "DaytonaNotFoundError", None)
        return isinstance(not_found, type) and isinstance(exc, not_found)

    def _require_sandbox(self) -> Any:
        if self._sandbox is None:
            raise SandboxError("Daytona sandbox is not open")
        return self._sandbox

    def _require_sdk(self) -> Any:
        if self._sdk is None:
            raise SandboxError("Daytona sandbox is not open")
        return self._sdk


def _load_daytona() -> Any:
    try:
        installed = version("daytona")
        sdk = import_module("daytona")
    except (ImportError, PackageNotFoundError) as exc:
        raise SandboxError("Install sqlsaber-sandbox[daytona] to use Daytona") from exc
    if not _daytona_version_supported(installed):
        raise SandboxError("Installed Daytona SDK is incompatible")
    required = (
        "AsyncDaytona",
        "CreateSandboxFromImageParams",
        "DaytonaConfig",
        "Image",
        "Resources",
        "SessionExecuteRequest",
        "DaytonaNotFoundError",
    )
    if any(not hasattr(sdk, name) for name in required):
        raise SandboxError("Installed Daytona SDK is incompatible")
    return sdk


def _daytona_version_supported(value: str) -> bool:
    match = re.match(r"^(\d+)\.(\d+)", value)
    return bool(match and int(match[1]) == 0 and int(match[2]) >= 143)


def _daytona_cpu(value: float | None) -> int | None:
    if value is None:
        return None
    numeric = float(value)
    if numeric < 1 or not numeric.is_integer():
        raise ValueError("Daytona cpu_cores must be a positive whole number")
    return int(numeric)


def _daytona_memory_gib(value: int | None) -> int | None:
    if value is None:
        return None
    gib, remainder = divmod(value, 1024)
    if gib < 1 or remainder:
        raise ValueError("Daytona memory_mb must be a whole number of GiB")
    return gib


def _daytona_gpu(value: str | None) -> int | None:
    if value is None:
        return None
    if not value.isdecimal() or int(value) < 1:
        raise ValueError("Daytona gpu must be a positive integer count")
    return int(value)


def _validate_timeout(timeout: float) -> None:
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError("timeout must be a finite positive number")
