"""Native E2B and Sprites backend contract tests without cloud resources."""

from __future__ import annotations

import asyncio
from inspect import signature
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from sqlsaber_sandbox.backends.base import CommandResult
from sqlsaber_sandbox.backends.e2b import E2BBackend
from sqlsaber_sandbox.backends.sprites import SpritesBackend
from sqlsaber_sandbox.config import SandboxConfig


class FakeE2BCommands:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.controller = object()

    async def run(self, command: str, **kwargs: object) -> object:
        from e2b import CommandExitException

        self.calls.append((command, kwargs))
        if kwargs.get("background"):
            return self.controller
        if command == "exit 7":
            raise CommandExitException(
                stdout="before failure\n",
                stderr="failed\n",
                exit_code=7,
                error=None,
            )
        return SimpleNamespace(stdout="ok\n", stderr="", exit_code=0)


class FakeE2BSandbox:
    def __init__(self) -> None:
        self.commands = FakeE2BCommands()
        self.files = SimpleNamespace(
            write=AsyncMock(),
            read=AsyncMock(return_value=bytearray(b"\x00\xffpayload")),
        )
        self.kill = AsyncMock()


async def test_e2b_uses_explicit_lifetime_binary_io_and_background_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from e2b import AsyncSandbox

    sandbox = FakeE2BSandbox()
    create = AsyncMock(return_value=sandbox)
    monkeypatch.setattr(AsyncSandbox, "create", create)
    backend = E2BBackend()

    await backend.open(SandboxConfig(provider="e2b", image="analysis-template"))
    assert await backend.execute("printf ok", timeout=1.25) == CommandResult(
        stdout="ok\n", stderr="", exit_code=0
    )
    assert await backend.execute("exit 7", timeout=2.5) == CommandResult(
        stdout="before failure\n", stderr="failed\n", exit_code=7
    )
    await backend.upload(b"\x00\xffpayload", "/tmp/value.bin")
    assert await backend.download("/tmp/value.bin") == b"\x00\xffpayload"
    await backend.start_controller(("python", "/tmp/controller script.py", "serve"))

    create.assert_awaited_once_with(template="analysis-template", timeout=3_600)
    sandbox.files.write.assert_awaited_once_with("/tmp/value.bin", b"\x00\xffpayload")
    sandbox.files.read.assert_awaited_once_with("/tmp/value.bin", format="bytes")
    controller_command, controller_options = sandbox.commands.calls[-1]
    assert controller_command == "python '/tmp/controller script.py' serve"
    assert controller_options["background"] is True
    assert controller_options["timeout"] == 0
    assert callable(controller_options["on_stdout"])
    assert callable(controller_options["on_stderr"])
    assert backend._controller is sandbox.commands.controller

    await backend.close()
    sandbox.kill.assert_awaited_once_with()


@pytest.mark.parametrize(
    "resource",
    [
        {"cpu_cores": 2, "image": "resource-template"},
        {"memory_mb": 2048},
        {"gpu": "T4"},
    ],
)
async def test_e2b_rejects_create_time_resources_even_with_a_template(
    resource: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="configured in a prebuilt template"):
        await E2BBackend().open(
            SandboxConfig(provider="e2b", **resource)  # ty: ignore[invalid-argument-type]
        )


async def test_e2b_close_keeps_failed_deletion_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from e2b import AsyncSandbox

    sandbox = FakeE2BSandbox()
    sandbox.kill.side_effect = [RuntimeError("temporary"), True]
    create = AsyncMock(return_value=sandbox)
    monkeypatch.setattr(AsyncSandbox, "create", create)
    backend = E2BBackend()
    await backend.open(SandboxConfig(provider="e2b", max_lifetime_seconds=1234))

    with pytest.raises(RuntimeError, match="temporary"):
        await backend.close()
    await backend.close()
    await backend.close()

    create.assert_awaited_once_with(template=None, timeout=1234)
    assert sandbox.kill.await_count == 2


class FakeSpritePath:
    def __init__(self, data: bytes = b"") -> None:
        self.data = data
        self.writes: list[bytes] = []

    async def write_bytes(self, data: bytes) -> None:
        self.data = data
        self.writes.append(data)

    async def read_bytes(self) -> bytes:
        return self.data


class FakeSpriteFilesystem:
    def __init__(self) -> None:
        self.paths: dict[str, FakeSpritePath] = {}

    def path(self, path: str) -> FakeSpritePath:
        return self.paths.setdefault(path, FakeSpritePath())


class FakeSpriteCommand:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.finished = asyncio.Event()

    async def run(self) -> None:
        self.started.set()
        try:
            await asyncio.Future()
        finally:
            self.finished.set()


class FakeSprite:
    def __init__(self) -> None:
        self.fs = FakeSpriteFilesystem()
        self.run = AsyncMock(
            return_value=SimpleNamespace(
                stdout=b"result:\xff\n",
                stderr=b"warning\n",
                returncode=9,
            )
        )
        self.destroy = AsyncMock()
        self.command_instance = FakeSpriteCommand()
        self.command_call: tuple[tuple[str, ...], dict[str, object]] | None = None
        self.filesystem_calls = 0

    def filesystem(self) -> FakeSpriteFilesystem:
        self.filesystem_calls += 1
        return self.fs

    def command(self, *argv: str, **kwargs: object) -> FakeSpriteCommand:
        self.command_call = (argv, kwargs)
        return self.command_instance


class FakeSpritesClient:
    def __init__(self, sprite: FakeSprite) -> None:
        self.create_sprite = AsyncMock(return_value=sprite)
        self.delete_sprite = AsyncMock()
        self.aclose = AsyncMock()


def install_fake_sprites_client(
    monkeypatch: pytest.MonkeyPatch,
    client: FakeSpritesClient,
) -> list[tuple[str, float]]:
    import sprites

    created_with: list[tuple[str, float]] = []

    def create_client(*, token: str, timeout: float) -> FakeSpritesClient:
        created_with.append((token, timeout))
        return client

    monkeypatch.setenv("SPRITES_TOKEN", "test-token")
    monkeypatch.setattr(sprites, "AsyncSpritesClient", create_client)
    return created_with


async def test_sprites_uses_native_resources_commands_and_binary_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sprite = FakeSprite()
    client = FakeSpritesClient(sprite)
    created_with = install_fake_sprites_client(monkeypatch, client)
    backend = SpritesBackend()

    await backend.open(
        SandboxConfig(
            provider="sprites",
            cpu_cores=2.0,
            memory_mb=4096,
            transport_seconds=17,
            max_lifetime_seconds=600,
        )
    )
    result = await backend.execute("printf result", timeout=4.5)
    await backend.upload(b"\x00\xffbinary", "/work/value.bin")
    assert await backend.download("/work/value.bin") == b"\x00\xffbinary"

    assert created_with == [("test-token", 17)]
    create_call = client.create_sprite.await_args
    assert create_call is not None
    name = create_call.args[0]
    resources = create_call.kwargs["config"]
    assert name.startswith("sqlsaber-")
    assert resources.ram_mb == 4096
    assert resources.cpus == 2
    assert result == CommandResult(
        stdout="result:\ufffd\n",
        stderr="warning\n",
        exit_code=9,
    )
    sprite.run.assert_awaited_once_with(
        "/bin/sh",
        "-lc",
        "printf result",
        capture_output=True,
        timeout=4.5,
    )
    assert sprite.filesystem_calls == 2
    assert sprite.fs.path("/work/value.bin").writes == [b"\x00\xffbinary"]

    await backend.close()
    sprite.destroy.assert_awaited_once_with()
    client.aclose.assert_awaited_once_with()


@pytest.mark.parametrize(
    ("resource", "message"),
    [
        ({"image": "custom"}, "image"),
        ({"gpu": "T4"}, "gpu"),
        ({"cpu_cores": 1.5}, "whole number"),
    ],
)
async def test_sprites_rejects_unsupported_resources(
    resource: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        await SpritesBackend().open(
            SandboxConfig(provider="sprites", **resource)  # ty: ignore[invalid-argument-type]
        )


async def test_sprites_controller_is_owned_drained_and_kept_on_failed_destroy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sprites import NotFoundError

    sprite = FakeSprite()
    sprite.destroy.side_effect = [
        RuntimeError("temporary"),
        NotFoundError("already deleted"),
    ]
    client = FakeSpritesClient(sprite)
    install_fake_sprites_client(monkeypatch, client)
    backend = SpritesBackend()
    await backend.open(SandboxConfig(provider="sprites"))

    await backend.start_controller(("python", "/work/controller.py", "serve"))
    await asyncio.wait_for(sprite.command_instance.started.wait(), timeout=1)
    assert sprite.command_call is not None
    argv, options = sprite.command_call
    assert argv == ("python", "/work/controller.py", "serve")
    assert options["timeout"] is None
    stdout = cast(Any, options["stdout"])
    stderr = cast(Any, options["stderr"])
    assert stdout.write(b"stdout") == 6
    assert stderr.write(b"stderr") == 6

    with pytest.raises(RuntimeError, match="temporary"):
        await backend.close()
    assert not sprite.command_instance.finished.is_set()
    client.aclose.assert_not_awaited()

    await backend.close()
    assert sprite.command_instance.finished.is_set()
    assert sprite.destroy.await_count == 2
    client.aclose.assert_awaited_once_with()


async def test_sprites_partial_open_and_client_close_are_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sprite = FakeSprite()
    client = FakeSpritesClient(sprite)
    client.create_sprite.side_effect = RuntimeError("capacity")
    client.aclose.side_effect = [RuntimeError("temporary"), None]
    install_fake_sprites_client(monkeypatch, client)
    backend = SpritesBackend()

    with pytest.raises(RuntimeError, match="capacity"):
        await backend.open(SandboxConfig(provider="sprites"))
    with pytest.raises(RuntimeError, match="temporary"):
        await backend.close()
    await backend.close()
    await backend.close()

    client.delete_sprite.assert_awaited_once()
    assert client.delete_sprite.await_args.args[0].startswith("sqlsaber-")
    assert client.aclose.await_count == 2


def test_installed_sdk_signatures_cover_the_adapter_calls() -> None:
    from e2b import AsyncSandbox
    from e2b.sandbox_async.commands.command import Commands
    from e2b.sandbox_async.filesystem.filesystem import Filesystem
    from sprites import AsyncSpritesClient
    from sprites.async_sprite import AsyncSprite

    assert {"template", "timeout"} <= signature(AsyncSandbox.create).parameters.keys()
    assert {"background", "timeout"} <= signature(Commands.run).parameters.keys()
    assert "format" in signature(Filesystem.read).parameters
    assert {"config", "name"} <= signature(
        AsyncSpritesClient.create_sprite
    ).parameters.keys()
    assert {"capture_output", "timeout"} <= signature(AsyncSprite.run).parameters.keys()
    assert {"stdout", "stderr", "timeout"} <= signature(
        AsyncSprite.command
    ).parameters.keys()
