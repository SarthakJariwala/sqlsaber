from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import Any

import pytest

from sqlsaber_notebook.execution import ExecutionLimits, NotebookInput
from sqlsaber_notebook.execution import modal as modal_backend


class AioMethod:
    def __init__(self, function: Callable[..., Awaitable[Any]]) -> None:
        self.aio = function


class FakeStream:
    def __init__(self, value: str) -> None:
        async def read() -> str:
            return value

        self.read = AioMethod(read)


class FakeProcess:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.stdout = FakeStream(stdout)
        self.stderr = FakeStream(stderr)

        async def wait() -> int:
            return returncode

        self.wait = AioMethod(wait)


class FakeFilesystem:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}

        async def make_directory(path: str, *, create_parents: bool = True) -> None:
            del path, create_parents

        async def write_bytes(data: bytes, path: str) -> None:
            self.files[path] = bytes(data)

        async def remove(path: str, *, recursive: bool = False) -> None:
            del path, recursive

        async def read_bytes(path: str) -> bytes:
            return self.files[path]

        self.make_directory = AioMethod(make_directory)
        self.write_bytes = AioMethod(write_bytes)
        self.remove = AioMethod(remove)
        self.read_bytes = AioMethod(read_bytes)


class FakeSandbox:
    def __init__(self) -> None:
        self.filesystem = FakeFilesystem()
        self.commands: list[tuple[str, ...]] = []
        self.terminated = False

        async def execute(*argv: str, **kwargs: Any) -> FakeProcess:
            del kwargs
            self.commands.append(argv)
            if argv[-2:] == ("nbconvert", "--version"):
                return FakeProcess(stdout="7.9.2\n")
            return FakeProcess()

        async def terminate(*, wait: bool = False) -> int:
            assert wait is True
            self.terminated = True
            return 137

        self.exec = AioMethod(execute)
        self.terminate = AioMethod(terminate)


async def test_modal_open_uses_token_credentials_without_environ_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = FakeSandbox()
    captured: dict[str, Any] = {}
    client_sentinel = object()

    async def from_credentials(token_id: str, token_secret: str) -> object:
        captured["credentials"] = (token_id, token_secret)
        return client_sentinel

    async def lookup(
        name: str,
        *,
        create_if_missing: bool,
        client: object | None = None,
    ) -> object:
        captured["lookup_client"] = client
        return object()

    async def create(*argv: str, **kwargs: Any) -> FakeSandbox:
        captured["create_client"] = kwargs.get("client")
        return sandbox

    fake_modal = SimpleNamespace(
        App=SimpleNamespace(lookup=AioMethod(lookup)),
        Client=SimpleNamespace(from_credentials=AioMethod(from_credentials)),
        Image=SimpleNamespace(from_registry=lambda image: f"image:{image}"),
        Sandbox=SimpleNamespace(create=AioMethod(create)),
    )
    monkeypatch.setattr(modal_backend, "_load_modal", lambda: fake_modal)
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)
    environ_before = dict(os.environ)

    backend = modal_backend.ModalNotebookBackend(
        token_id="ak-token",
        token_secret="as-secret",
    )
    environment = await backend.open(
        [NotebookInput("data.json", b"{}")],
        image="registry/image@sha256:digest",
        limits=ExecutionLimits(),
    )

    assert captured["credentials"] == ("ak-token", "as-secret")
    assert captured["lookup_client"] is client_sentinel
    assert captured["create_client"] is client_sentinel
    assert dict(os.environ) == environ_before
    await environment.close()


def test_modal_backend_requires_complete_token_pair() -> None:
    with pytest.raises(ValueError, match="token_id and token_secret together"):
        modal_backend.ModalNotebookBackend(token_id="ak-token")
    with pytest.raises(ValueError, match="token_id and token_secret together"):
        modal_backend.ModalNotebookBackend(token_secret="as-secret")


async def test_modal_open_uses_direct_image_blocked_network_and_non_root_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = FakeSandbox()
    captured: dict[str, Any] = {}

    async def lookup(name: str, *, create_if_missing: bool) -> object:
        captured["app"] = (name, create_if_missing)
        return object()

    async def create(*argv: str, **kwargs: Any) -> FakeSandbox:
        captured["create"] = (argv, kwargs)
        return sandbox

    def from_registry(image: str) -> str:
        captured["image"] = image
        return f"image:{image}"

    fake_modal = SimpleNamespace(
        App=SimpleNamespace(lookup=AioMethod(lookup)),
        Image=SimpleNamespace(from_registry=from_registry),
        Sandbox=SimpleNamespace(create=AioMethod(create)),
    )
    monkeypatch.setattr(modal_backend, "_load_modal", lambda: fake_modal)

    backend = modal_backend.ModalNotebookBackend()
    environment = await backend.open(
        [NotebookInput("data.json", b"{}")],
        image="registry/image@sha256:digest",
        limits=ExecutionLimits(),
    )

    assert captured["app"] == ("sqlsaber-notebook", True)
    assert captured["image"] == "registry/image@sha256:digest"
    argv, kwargs = captured["create"]
    assert argv == ("sleep", "infinity")
    assert kwargs["block_network"] is True
    assert kwargs["cpu"] == 4.0
    assert kwargs["memory"] == 8192
    assert kwargs["timeout"] == 86_400
    assert "idle_timeout" not in kwargs
    assert "experimental_options" not in kwargs
    assert any(command[0] == "chmod" for command in sandbox.commands)
    assert any(
        command[:5] == ("/usr/sbin/runuser", "-u", "jovyan", "--", "jupyter")
        for command in sandbox.commands
    )
    assert sandbox.filesystem.files[f"{environment.inputs_path}/data.json"] == b"{}"

    await environment.close()
    await environment.close()
    assert sandbox.terminated is True
