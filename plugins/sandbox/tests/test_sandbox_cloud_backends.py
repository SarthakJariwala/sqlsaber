"""Native Modal and Daytona backend contract tests without cloud resources."""

from __future__ import annotations

import asyncio
import inspect
import os
import shlex
from types import SimpleNamespace
from typing import Any

import pytest

from sqlsaber_sandbox.backends import daytona as daytona_backend
from sqlsaber_sandbox.backends import modal as modal_backend
from sqlsaber_sandbox.backends.base import CommandResult, SandboxError
from sqlsaber_sandbox.config import SandboxConfig

SCIPY_IMAGE = (
    "quay.io/jupyter/scipy-notebook@sha256:"
    "e6e8baae46b5e62bbc26910169082639a6fd96f90e9f6fc52e0c0389df92d35c"
)


class AioMethod:
    def __init__(self, function: Any) -> None:
        self.aio = function


def cloud_config(
    *,
    image: str | None = None,
    cpu_cores: float | None = None,
    memory_mb: int | None = None,
    gpu: str | None = None,
    open_seconds: int = 17,
    transport_seconds: int = 7,
    idle_seconds: float | None = None,
    max_lifetime_seconds: int | None = None,
) -> SandboxConfig:
    return SandboxConfig(
        image=image,
        cpu_cores=cpu_cores,
        memory_mb=memory_mb,
        gpu=gpu,
        open_seconds=open_seconds,
        transport_seconds=transport_seconds,
        idle_seconds=idle_seconds,
        max_lifetime_seconds=max_lifetime_seconds,
    )


class FakeModalStream:
    def __init__(self, value: str, finished: asyncio.Event | None = None) -> None:
        async def read() -> str:
            if finished is not None:
                await finished.wait()
            return value

        self.read = AioMethod(read)


class FakeModalProcess:
    def __init__(
        self,
        *,
        stdout: str = "out\n",
        stderr: str = "err\n",
        exit_code: int = 23,
        finished: asyncio.Event | None = None,
    ) -> None:
        self.stdout = FakeModalStream(stdout, finished)
        self.stderr = FakeModalStream(stderr, finished)

        async def wait() -> int:
            if finished is not None:
                await finished.wait()
            return exit_code

        self.wait = AioMethod(wait)


class FakeModalFilesystem:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}

        async def write_bytes(data: bytes, path: str) -> None:
            self.files[path] = bytes(data)

        async def read_bytes(path: str) -> bytes:
            return self.files[path]

        self.write_bytes = AioMethod(write_bytes)
        self.read_bytes = AioMethod(read_bytes)


class FakeModalSandbox:
    def __init__(self) -> None:
        self.filesystem = FakeModalFilesystem()
        self.exec_calls: list[tuple[tuple[str, ...], dict[str, Any]]] = []
        self.lifecycle: list[str] = []
        self.controller_finished = asyncio.Event()
        self.detach_failures = 0

        async def execute(*argv: str, **kwargs: Any) -> FakeModalProcess:
            self.exec_calls.append((argv, kwargs))
            if argv and argv[0] == "python":
                return FakeModalProcess(
                    stdout="controller out",
                    stderr="controller err",
                    exit_code=137,
                    finished=self.controller_finished,
                )
            return FakeModalProcess()

        async def terminate(*, wait: bool) -> int:
            assert wait is True
            self.lifecycle.append("terminate")
            self.controller_finished.set()
            return 137

        async def detach() -> None:
            self.lifecycle.append("detach")
            if self.detach_failures:
                self.detach_failures -= 1
                raise RuntimeError("MODAL_TOKEN_SECRET=should-not-leak")

        self.exec = AioMethod(execute)
        self.terminate = AioMethod(terminate)
        self.detach = AioMethod(detach)


def fake_modal_sdk(
    sandbox: FakeModalSandbox,
    *,
    create_error: Exception | None = None,
) -> tuple[Any, dict[str, Any]]:
    captured: dict[str, Any] = {}

    class FakeModalClient:
        pass

    async def from_credentials(token_id: str, token_secret: str) -> FakeModalClient:
        captured["credentials"] = (token_id, token_secret)
        client = FakeModalClient()
        captured["client"] = client
        return client

    async def lookup(
        name: str, *, create_if_missing: bool, client: object | None = None
    ) -> object:
        captured["lookup"] = (name, create_if_missing)
        captured["lookup_client"] = client
        return "app"

    async def create(*argv: str, **kwargs: Any) -> FakeModalSandbox:
        captured["create"] = (argv, kwargs)
        if create_error is not None:
            raise create_error
        return sandbox

    def debian_slim(*, python_version: str) -> str:
        captured["default_python"] = python_version
        return "default-image"

    def from_registry(image: str) -> str:
        captured["registry_image"] = image
        return f"registry:{image}"

    sdk = SimpleNamespace(
        App=SimpleNamespace(lookup=AioMethod(lookup)),
        Client=SimpleNamespace(from_credentials=AioMethod(from_credentials)),
        Image=SimpleNamespace(
            debian_slim=debian_slim,
            from_registry=from_registry,
        ),
        Sandbox=SimpleNamespace(create=AioMethod(create)),
    )
    return sdk, captured


async def test_modal_native_mapping_execution_transfer_and_controller_lifetime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = FakeModalSandbox()
    sdk, captured = fake_modal_sdk(sandbox)
    monkeypatch.setattr(modal_backend, "_load_modal", lambda: sdk)
    backend = modal_backend.ModalBackend()

    await backend.open(
        cloud_config(
            cpu_cores=2.5,
            memory_mb=4096,
            gpu="A10G",
            idle_seconds=91.2,
        )
    )

    assert captured["lookup"] == ("sqlsaber-sandbox", True)
    assert captured["registry_image"] == SCIPY_IMAGE
    argv, options = captured["create"]
    assert argv == ("sleep", "infinity")
    assert options == {
        "app": "app",
        "image": f"registry:{SCIPY_IMAGE}",
        "timeout": 86_400,
        "cpu": 2.5,
        "memory": 4096,
        "gpu": "A10G",
    }

    result = await backend.execute("printf test", timeout=4.2)
    assert result == CommandResult(stdout="out\n", stderr="err\n", exit_code=23)
    assert sandbox.exec_calls[-1] == (
        ("sh", "-lc", "printf test"),
        {"timeout": 5},
    )

    binary = bytes(range(256)) + b"\x00\xfftail"
    await backend.upload(binary, "/tmp/input.bin")
    assert await backend.download("/tmp/input.bin") == binary

    controller_argv = ("python", "/tmp/controller.py", "space value")
    await backend.start_controller(controller_argv)
    await asyncio.sleep(0)
    assert sandbox.exec_calls[-1] == (controller_argv, {})
    assert backend._controller_process is not None
    assert any(not task.done() for task in backend._controller_tasks)
    controller_tasks = backend._controller_tasks

    await backend.close()
    await backend.close()
    assert sandbox.lifecycle == ["terminate", "detach"]
    assert backend._sandbox is None
    assert all(task.done() for task in controller_tasks)


async def test_modal_custom_image_lifetime_and_retryable_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = FakeModalSandbox()
    sandbox.detach_failures = 1
    sdk, captured = fake_modal_sdk(sandbox)
    monkeypatch.setattr(modal_backend, "_load_modal", lambda: sdk)
    backend = modal_backend.ModalBackend()
    await backend.open(cloud_config(image="python:custom", max_lifetime_seconds=600))

    assert captured["registry_image"] == "python:custom"
    assert captured["create"][1]["timeout"] == 600
    with pytest.raises(SandboxError, match="Could not close Modal sandbox") as raised:
        await backend.close()
    assert "MODAL_TOKEN_SECRET" not in str(raised.value)
    assert backend._sandbox is sandbox

    await backend.close()
    assert sandbox.lifecycle == ["terminate", "detach", "terminate", "detach"]
    assert backend._sandbox is None


async def test_modal_passes_explicit_client_to_lookup_and_create(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODAL_TOKEN_ID", "unchanged-native-id")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "unchanged-native-secret")
    sandbox = FakeModalSandbox()
    sdk, captured = fake_modal_sdk(sandbox)
    monkeypatch.setattr(modal_backend, "_load_modal", lambda: sdk)
    backend = modal_backend.ModalBackend(
        token_id="saved-modal-id",
        token_secret="saved-modal-secret",
    )

    await backend.open(cloud_config())

    client = captured["client"]
    assert captured["credentials"] == ("saved-modal-id", "saved-modal-secret")
    assert captured["lookup"] == ("sqlsaber-sandbox", True)
    assert captured["lookup_client"] is client
    assert captured["create"][1]["client"] is client
    assert os.environ["MODAL_TOKEN_ID"] == "unchanged-native-id"
    assert os.environ["MODAL_TOKEN_SECRET"] == "unchanged-native-secret"
    await backend.close()


async def test_modal_rejects_unsupported_lifetime_before_loading_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected() -> Any:
        raise AssertionError("SDK should not be loaded")

    monkeypatch.setattr(modal_backend, "_load_modal", unexpected)
    with pytest.raises(ValueError, match="cannot exceed 86400"):
        await modal_backend.ModalBackend().open(
            cloud_config(max_lifetime_seconds=86_401)
        )


async def test_modal_create_failure_is_safe_and_close_handles_partial_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk, captured = fake_modal_sdk(
        FakeModalSandbox(),
        create_error=RuntimeError("MODAL_TOKEN_SECRET=[REDACTED:secret-value]"),
    )
    monkeypatch.setattr(modal_backend, "_load_modal", lambda: sdk)
    backend = modal_backend.ModalBackend()

    with pytest.raises(SandboxError, match="Could not open Modal sandbox") as raised:
        await backend.open(cloud_config())
    assert "MODAL_TOKEN_SECRET" not in str(raised.value)
    assert captured["lookup"] == ("sqlsaber-sandbox", True)
    assert backend._sandbox is None
    await backend.close()


class DaytonaNotFoundError(RuntimeError):
    pass


class FakeDaytonaResources:
    def __init__(
        self,
        cpu: int | None = None,
        memory: int | None = None,
        disk: int | None = None,
        gpu: int | None = None,
    ) -> None:
        self.cpu = cpu
        self.memory = memory
        self.disk = disk
        self.gpu = gpu


class FakeDaytonaParams:
    def __init__(self, **values: Any) -> None:
        self.__dict__.update(values)


class FakeDaytonaConfig:
    def __init__(self, **values: Any) -> None:
        self.__dict__.update(values)


class FakeDaytonaImage:
    versions: list[str] = []

    @classmethod
    def debian_slim(cls, version: str) -> str:
        cls.versions.append(version)
        return f"python:{version}"


class FakeSessionRequest:
    def __init__(self, *, command: str, run_async: bool) -> None:
        self.command = command
        self.run_async = run_async


class FakeDaytonaProcess:
    def __init__(self) -> None:
        self.sessions: set[str] = set()
        self.calls: list[tuple[str, FakeSessionRequest, int | None]] = []
        self.deleted_sessions: list[str] = []

    async def create_session(self, session_id: str) -> None:
        self.sessions.add(session_id)

    async def execute_session_command(
        self,
        session_id: str,
        request: FakeSessionRequest,
        timeout: int | None = None,
    ) -> Any:
        assert session_id in self.sessions
        self.calls.append((session_id, request, timeout))
        if request.run_async:
            return SimpleNamespace(
                cmd_id="controller-command",
                stdout=None,
                stderr=None,
                exit_code=None,
            )
        return SimpleNamespace(
            cmd_id="command",
            stdout="separate stdout\n",
            stderr="separate stderr\n",
            exit_code=19,
        )

    async def delete_session(self, session_id: str) -> None:
        self.deleted_sessions.append(session_id)
        self.sessions.discard(session_id)


class FakeDaytonaFilesystem:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}

    async def upload_file(
        self,
        data: bytes,
        path: str,
        timeout: int,
    ) -> None:
        assert timeout == 7
        self.files[path] = bytes(data)

    async def download_file(self, path: str, timeout: int) -> bytes:
        assert timeout == 7
        return self.files[path]


class FakeDaytonaSandbox:
    def __init__(self, client: FakeDaytonaClient) -> None:
        self.client = client
        self.process = FakeDaytonaProcess()
        self.fs = FakeDaytonaFilesystem()
        self.delete_calls = 0
        self.delete_errors: list[Exception] = []

    async def delete(self, timeout: float | None = 60) -> None:
        assert timeout == 60
        self.delete_calls += 1
        if self.delete_errors:
            raise self.delete_errors.pop(0)
        self.client.deleted = True


class FakeDaytonaClient:
    def __init__(self, sdk: Any) -> None:
        self.sdk = sdk
        self.params: FakeDaytonaParams | None = None
        self.create_timeout: float | None = None
        self.sandbox: FakeDaytonaSandbox | None = None
        self.deleted = False
        self.closed = False
        self.close_errors: list[Exception] = []
        sdk.clients.append(self)

    async def create(
        self,
        params: FakeDaytonaParams,
        *,
        timeout: float,
    ) -> FakeDaytonaSandbox:
        self.params = params
        self.create_timeout = timeout
        self.sandbox = FakeDaytonaSandbox(self)
        if self.sdk.create_error is not None:
            raise self.sdk.create_error
        return self.sandbox

    async def get(self, name: str) -> FakeDaytonaSandbox:
        assert self.params is not None
        assert name == getattr(self.params, "name")
        if self.sandbox is None or self.deleted:
            raise DaytonaNotFoundError(name)
        return self.sandbox

    async def close(self) -> None:
        if self.close_errors:
            raise self.close_errors.pop(0)
        self.closed = True


def fake_daytona_sdk() -> Any:
    FakeDaytonaImage.versions = []
    sdk = SimpleNamespace(
        clients=[],
        configs=[],
        create_error=None,
        AsyncDaytona=None,
        CreateSandboxFromImageParams=FakeDaytonaParams,
        DaytonaConfig=FakeDaytonaConfig,
        Image=FakeDaytonaImage,
        Resources=FakeDaytonaResources,
        SessionExecuteRequest=FakeSessionRequest,
        DaytonaNotFoundError=DaytonaNotFoundError,
    )

    def create_client(config: FakeDaytonaConfig | None = None) -> FakeDaytonaClient:
        sdk.configs.append(config)
        return FakeDaytonaClient(sdk)

    sdk.AsyncDaytona = create_client
    return sdk


async def test_daytona_native_mapping_execution_transfer_and_controller_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = fake_daytona_sdk()
    monkeypatch.setattr(daytona_backend, "_load_daytona", lambda: sdk)
    backend = daytona_backend.DaytonaBackend()
    await backend.open(
        cloud_config(
            cpu_cores=2,
            memory_mb=2048,
            gpu="2",
            idle_seconds=61,
        )
    )

    client = sdk.clients[0]
    params = client.params
    sandbox = client.sandbox
    assert params is not None
    assert sandbox is not None
    assert FakeDaytonaImage.versions == []
    assert params.image == SCIPY_IMAGE
    assert params.language == "python"
    assert params.name.startswith("sqlsaber-sandbox-")
    assert params.labels == {
        "application": "sqlsaber",
        "purpose": "sandbox-analysis",
    }
    assert params.resources.cpu == 2
    assert params.resources.memory == 2
    assert params.resources.gpu == 2
    assert params.auto_stop_interval == 0
    assert params.ephemeral is True
    assert client.create_timeout == 17
    assert sdk.configs == [None]

    result = await backend.execute("printf mixed", timeout=3.1)
    assert result == CommandResult(
        stdout="separate stdout\n",
        stderr="separate stderr\n",
        exit_code=19,
    )
    transient_id, request, command_timeout = sandbox.process.calls[-1]
    assert request.command == "printf mixed"
    assert request.run_async is False
    assert command_timeout == 4
    assert transient_id in sandbox.process.deleted_sessions

    binary = bytes(range(255, -1, -1)) + b"\x00\xfftail"
    await backend.upload(binary, "/tmp/data.bin")
    assert await backend.download("/tmp/data.bin") == binary

    controller_argv = ("python", "/tmp/controller.py", "space value")
    await backend.start_controller(controller_argv)
    controller_id, request, controller_timeout = sandbox.process.calls[-1]
    assert request.command == shlex.join(controller_argv)
    assert request.run_async is True
    assert controller_timeout == 7
    assert controller_id in sandbox.process.sessions
    assert backend._controller_session_id == controller_id
    assert backend._controller_command_id == "controller-command"

    await backend.close()
    await backend.close()
    assert sandbox.delete_calls == 1
    assert client.closed is True
    assert backend._sandbox is None
    assert backend._client is None


async def test_daytona_custom_image_disables_hidden_idle_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = fake_daytona_sdk()
    monkeypatch.setattr(daytona_backend, "_load_daytona", lambda: sdk)
    backend = daytona_backend.DaytonaBackend()

    await backend.open(cloud_config(image="python:custom"))
    client = sdk.clients[0]
    assert client.params is not None
    assert client.params.image == "python:custom"
    assert client.params.auto_stop_interval == 0
    assert FakeDaytonaImage.versions == []
    await backend.close()


async def test_daytona_passes_injected_key_and_endpoint_to_client_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DAYTONA_API_KEY", raising=False)
    monkeypatch.delenv("DAYTONA_API_URL", raising=False)
    sdk = fake_daytona_sdk()
    monkeypatch.setattr(daytona_backend, "_load_daytona", lambda: sdk)
    backend = daytona_backend.DaytonaBackend(
        api_key="saved-daytona-key",
        api_url="https://daytona.example/api",
    )

    await backend.open(cloud_config())

    assert len(sdk.configs) == 1
    assert sdk.configs[0].api_key == "saved-daytona-key"
    assert sdk.configs[0].api_url == "https://daytona.example/api"
    assert "DAYTONA_API_KEY" not in os.environ
    assert "DAYTONA_API_URL" not in os.environ
    await backend.close()


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"cpu_cores": 1.5}, "whole number"),
        ({"memory_mb": 1536}, "whole number of GiB"),
        ({"gpu": "A10G"}, "positive integer count"),
    ],
)
async def test_daytona_rejects_unrepresentable_config_before_loading_sdk(
    monkeypatch: pytest.MonkeyPatch,
    changes: dict[str, Any],
    message: str,
) -> None:
    def unexpected() -> Any:
        raise AssertionError("SDK should not be loaded")

    monkeypatch.setattr(daytona_backend, "_load_daytona", unexpected)
    with pytest.raises(ValueError, match=message):
        await daytona_backend.DaytonaBackend().open(cloud_config(**changes))


async def test_daytona_failed_delete_and_client_close_are_independently_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = fake_daytona_sdk()
    monkeypatch.setattr(daytona_backend, "_load_daytona", lambda: sdk)
    backend = daytona_backend.DaytonaBackend()
    await backend.open(cloud_config())
    client = sdk.clients[0]
    sandbox = client.sandbox
    assert sandbox is not None
    sandbox.delete_errors.append(RuntimeError("DAYTONA_API_KEY=should-not-leak"))

    with pytest.raises(SandboxError, match="Could not close Daytona sandbox") as raised:
        await backend.close()
    assert "DAYTONA_API_KEY" not in str(raised.value)
    assert backend._sandbox is sandbox
    assert client.closed is False

    client.close_errors.append(RuntimeError("temporary client failure"))
    with pytest.raises(SandboxError, match="Could not close Daytona client"):
        await backend.close()
    assert backend._sandbox is None
    assert backend._client is client
    assert sandbox.delete_calls == 2

    await backend.close()
    assert client.closed is True
    assert sandbox.delete_calls == 2


async def test_daytona_create_failure_recovers_and_deletes_partial_sandbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = fake_daytona_sdk()
    sdk.create_error = RuntimeError("DAYTONA_API_KEY=should-not-leak")
    monkeypatch.setattr(daytona_backend, "_load_daytona", lambda: sdk)
    backend = daytona_backend.DaytonaBackend()

    with pytest.raises(SandboxError, match="Could not open Daytona sandbox") as raised:
        await backend.open(cloud_config())
    assert "DAYTONA_API_KEY" not in str(raised.value)
    client = sdk.clients[0]
    assert client.sandbox is not None
    assert client.sandbox.delete_calls == 1
    assert client.closed is True
    assert backend._sandbox is None
    assert backend._client is None


def test_installed_cloud_sdk_signatures_cover_the_native_calls() -> None:
    modal = modal_backend._load_modal()
    modal_create = inspect.signature(modal.Sandbox.create.aio).parameters
    assert {"timeout", "idle_timeout", "cpu", "memory", "gpu"} <= modal_create.keys()
    assert hasattr(modal.App.lookup, "aio")
    assert hasattr(modal.Client.from_credentials, "aio")
    assert "client" in inspect.signature(modal.App.lookup.aio).parameters
    assert "client" in modal_create

    from modal.sandbox_fs import SandboxFilesystem

    assert hasattr(SandboxFilesystem.write_bytes, "aio")
    assert hasattr(SandboxFilesystem.read_bytes, "aio")

    daytona = daytona_backend._load_daytona()
    create_fields = inspect.signature(daytona.CreateSandboxFromImageParams).parameters
    assert {"image", "resources", "auto_stop_interval", "ephemeral"} <= (
        create_fields.keys()
    )
    assert {"api_key", "api_url"} <= inspect.signature(
        daytona.DaytonaConfig
    ).parameters.keys()
    assert "ttl_minutes" not in create_fields
    assert {"cpu", "memory", "gpu"} <= inspect.signature(
        daytona.Resources
    ).parameters.keys()
    assert "run_async" in inspect.signature(daytona.SessionExecuteRequest).parameters
