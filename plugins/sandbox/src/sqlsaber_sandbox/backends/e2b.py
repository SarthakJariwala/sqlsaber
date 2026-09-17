"""Native asynchronous E2B sandbox backend."""

from __future__ import annotations

import shlex
from collections.abc import Sequence
from typing import Any

from ..config import SandboxConfig
from .base import CommandResult, SandboxError, SessionLost

_HOBBY_LIFETIME_SECONDS = 3_600


class E2BBackend:
    """An E2B sandbox whose lifetime and background controller are owned here."""

    def __init__(self, *, api_key: str | None = None) -> None:
        self._api_key = api_key
        self._sandbox: Any | None = None
        self._controller: Any | None = None

    async def open(self, config: SandboxConfig) -> None:
        if self._sandbox is not None:
            raise SandboxError("E2B backend is already open")
        if (
            config.cpu_cores is not None
            or config.memory_mb is not None
            or config.gpu is not None
        ):
            raise ValueError(
                "E2B CPU, memory, and GPU resources must be configured in a prebuilt template, not as create-time options"
            )
        try:
            from e2b import AsyncSandbox
        except ImportError as exc:
            raise RuntimeError(
                "E2B support requires the sandbox extra (install the 'e2b' package)"
            ) from exc

        # An explicit lifetime avoids E2B's implicit 300-second default. E2B
        # documents 3,600 seconds as the maximum available on Hobby plans.
        lifetime = config.max_lifetime_seconds or _HOBBY_LIFETIME_SECONDS
        options: dict[str, Any] = {"template": config.image, "timeout": lifetime}
        if self._api_key is not None:
            options["api_key"] = self._api_key
        self._sandbox = await AsyncSandbox.create(**options)

    def _require_sandbox(self) -> Any:
        if self._sandbox is None:
            raise SessionLost("E2B backend is not open")
        return self._sandbox

    async def execute(self, command: str, *, timeout: float) -> CommandResult:
        sandbox = self._require_sandbox()
        from e2b import CommandExitException

        try:
            result = await sandbox.commands.run(command, timeout=timeout)
        except CommandExitException as exc:
            result = exc
        return CommandResult(result.stdout, result.stderr, result.exit_code)

    async def upload(self, data: bytes, path: str) -> None:
        await self._require_sandbox().files.write(path, data)

    async def download(self, path: str) -> bytes:
        data = await self._require_sandbox().files.read(path, format="bytes")
        return bytes(data)

    async def start_controller(self, argv: Sequence[str]) -> None:
        if not argv:
            raise ValueError("Controller argv cannot be empty")
        if self._controller is not None:
            raise SandboxError("Controller is already running")
        command = shlex.join(argv)
        self._controller = await self._require_sandbox().commands.run(
            command,
            background=True,
            timeout=0,
            # Drain both output streams rather than leaving a bounded SDK queue full.
            on_stdout=lambda _output: None,
            on_stderr=lambda _output: None,
        )

    async def close(self) -> None:
        sandbox = self._sandbox
        if sandbox is None:
            return
        await sandbox.kill()
        self._sandbox = None
        self._controller = None
