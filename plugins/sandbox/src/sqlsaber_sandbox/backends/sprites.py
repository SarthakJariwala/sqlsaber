"""Native asynchronous Sprites sandbox backend."""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import Sequence
from typing import Any

from ..config import SandboxConfig
from .base import CommandResult, SandboxError, SessionLost


class _DiscardBytes:
    def write(self, data: bytes) -> int:
        return len(data)


class SpritesBackend:
    """One Sprite with an owned, continuously drained controller command."""

    def __init__(self) -> None:
        self._client: Any | None = None
        self._sprite: Any | None = None
        self._name: str | None = None
        self._controller: Any | None = None
        self._controller_task: asyncio.Task[None] | None = None

    async def open(self, config: SandboxConfig) -> None:
        if self._sprite is not None or self._client is not None:
            raise SandboxError("Sprites backend is already open")
        unsupported = [
            name
            for name, value in (
                ("image", config.image),
                ("gpu", config.gpu),
            )
            if value is not None
        ]
        if unsupported:
            raise ValueError(
                "Sprites does not support these sandbox options: "
                + ", ".join(unsupported)
            )
        if config.cpu_cores is not None and not float(config.cpu_cores).is_integer():
            raise ValueError("Sprites cpu_cores must be a whole number")
        try:
            from sprites import AsyncSpritesClient, SpriteConfig
        except ImportError as exc:
            raise RuntimeError(
                "Sprites support requires the sandbox extra (install the 'sprites-py' package)"
            ) from exc

        token = os.getenv("SPRITES_TOKEN")
        if not token:
            raise RuntimeError("SPRITES_TOKEN is required to use the Sprites backend")
        client = AsyncSpritesClient(token=token, timeout=config.transport_seconds)
        self._client = client
        resources = SpriteConfig(
            cpus=int(config.cpu_cores) if config.cpu_cores is not None else None,
            ram_mb=config.memory_mb,
        )
        self._name = f"sqlsaber-{uuid.uuid4().hex}"
        self._sprite = await client.create_sprite(self._name, config=resources)

    def _require_sprite(self) -> Any:
        if self._sprite is None:
            raise SessionLost("Sprites backend is not open")
        return self._sprite

    async def execute(self, command: str, *, timeout: float) -> CommandResult:
        result = await self._require_sprite().run(
            "/bin/sh", "-lc", command, capture_output=True, timeout=timeout
        )
        return CommandResult(
            (result.stdout or b"").decode(errors="replace"),
            (result.stderr or b"").decode(errors="replace"),
            result.returncode,
        )

    async def upload(self, data: bytes, path: str) -> None:
        await self._require_sprite().filesystem().path(path).write_bytes(data)

    async def download(self, path: str) -> bytes:
        return await self._require_sprite().filesystem().path(path).read_bytes()

    async def start_controller(self, argv: Sequence[str]) -> None:
        if not argv:
            raise ValueError("Controller argv cannot be empty")
        if self._controller_task is not None:
            raise SandboxError("Controller is already running")

        controller = self._require_sprite().command(
            *argv,
            stdout=_DiscardBytes(),
            stderr=_DiscardBytes(),
            timeout=None,
        )
        task = asyncio.create_task(controller.run(), name="sqlsaber-sprites-controller")
        self._controller = controller
        self._controller_task = task
        await asyncio.sleep(0)
        if task.done():
            self._controller = None
            self._controller_task = None
            task.result()
            raise SandboxError("Controller exited during startup")

    async def close(self) -> None:
        if self._sprite is not None or self._name is not None:
            from sprites import NotFoundError

            try:
                if self._sprite is not None:
                    await self._sprite.destroy()
                elif self._client is not None:
                    await self._client.delete_sprite(self._name)
            except NotFoundError:
                pass
            self._sprite = None
            self._name = None
            await self._finish_controller()
        elif self._controller_task is not None:
            await self._finish_controller()
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _finish_controller(self) -> None:
        task = self._controller_task
        if task is not None:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self._controller = None
        self._controller_task = None
