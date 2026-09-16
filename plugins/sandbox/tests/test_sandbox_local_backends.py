from __future__ import annotations

import asyncio
import inspect
import os
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from sqlsaber_sandbox.backends.base import SandboxError
from sqlsaber_sandbox.backends import docker as docker_backend
from sqlsaber_sandbox.backends.docker import DockerBackend, _ProcessResult
from sqlsaber_sandbox.backends import microsandbox as microsandbox_backend
from sqlsaber_sandbox.backends.microsandbox import MicrosandboxBackend
from sqlsaber_sandbox.config import SandboxConfig


async def test_docker_honors_resources_without_host_mounts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[str, ...], bytes | None, float | None]] = []

    async def run(
        argv: Any, *, input_data: bytes | None = None, timeout: float | None = None
    ) -> _ProcessResult:
        calls.append((tuple(argv), input_data, timeout))
        return _ProcessResult(0, b"container\n", b"")

    monkeypatch.setattr(docker_backend, "_run_process", run)
    backend = DockerBackend(executable="docker")
    await backend.open(
        SandboxConfig(
            provider="docker",
            image="python:3.12-slim",
            cpu_cores=1.5,
            memory_mb=768,
            gpu="device=0",
        )
    )

    create = next(argv for argv, _, _ in calls if argv[1] == "create")
    assert create[create.index("--cpus") + 1] == "1.5"
    assert create[create.index("--memory") + 1] == "768m"
    assert create[create.index("--gpus") + 1] == "device=0"
    assert "--cap-drop" in create and "no-new-privileges" in create
    assert not {"--mount", "--volume", "-v"}.intersection(create)
    assert all("/home/" not in argument for argument in create)
    await backend.close()


async def test_docker_transfer_is_binary_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = bytes(range(256)) + b"\x00\xff\r\n"
    uploaded: list[bytes] = []

    async def run(
        argv: Any, *, input_data: bytes | None = None, timeout: float | None = None
    ) -> _ProcessResult:
        del timeout
        arguments = tuple(argv)
        if input_data is not None:
            uploaded.append(input_data)
        if "sqlsaber-download" in arguments:
            return _ProcessResult(0, payload, b"")
        return _ProcessResult(0, b"ok", b"")

    monkeypatch.setattr(docker_backend, "_run_process", run)
    backend = DockerBackend(executable="docker")
    await backend.open(SandboxConfig(provider="docker", image="python:3.12"))
    await backend.upload(payload, "/tmp/data.bin")
    assert uploaded == [payload]
    assert await backend.download("/tmp/data.bin") == payload
    await backend.close()


async def test_docker_uses_default_image_when_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []

    async def run(
        argv: Any, *, input_data: bytes | None = None, timeout: float | None = None
    ) -> _ProcessResult:
        del input_data, timeout
        calls.append(tuple(argv))
        return _ProcessResult(0, b"ok", b"")

    monkeypatch.setattr(docker_backend, "_run_process", run)
    backend = DockerBackend(executable="docker")
    await backend.open(SandboxConfig(provider="docker"))
    create = next(argv for argv in calls if argv[1] == "create")
    assert create[-4:] == (
        "quay.io/jupyter/scipy-notebook@sha256:"
        "e6e8baae46b5e62bbc26910169082639a6fd96f90e9f6fc52e0c0389df92d35c",
        "python",
        "-c",
        "import time; time.sleep(2147483647)",
    )
    await backend.close()


class _BlockingStream:
    def __init__(self) -> None:
        self.closed = asyncio.Event()

    async def read(self, size: int = -1) -> bytes:
        del size
        await self.closed.wait()
        return b""


class _ControllerProcess:
    def __init__(self) -> None:
        self.returncode: int | None = None
        self.stdout = _BlockingStream()
        self.stderr = _BlockingStream()
        self.stopped = asyncio.Event()
        self.terminated = False
        self.killed = False

    async def wait(self) -> int:
        await self.stopped.wait()
        assert self.returncode is not None
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15
        self.stdout.closed.set()
        self.stderr.closed.set()
        self.stopped.set()

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9
        self.stdout.closed.set()
        self.stderr.closed.set()
        self.stopped.set()


async def test_docker_controller_stays_attached_across_exec_and_is_drained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _ControllerProcess()

    async def run(
        argv: Any, *, input_data: bytes | None = None, timeout: float | None = None
    ) -> _ProcessResult:
        del input_data, timeout
        arguments = tuple(argv)
        if arguments[-3:] == ("sh", "-lc", "printf later"):
            return _ProcessResult(0, b"later", b"")
        return _ProcessResult(0, b"ok", b"")

    async def spawn(argv: Any, **kwargs: Any) -> Any:
        del argv, kwargs
        return process

    monkeypatch.setattr(docker_backend, "_run_process", run)
    monkeypatch.setattr(docker_backend, "_spawn_process", spawn)
    backend = DockerBackend(executable="docker")
    await backend.open(SandboxConfig(provider="docker", image="python:3.12"))
    await backend.start_controller(("python", "/tmp/controller.py"))
    assert process.returncode is None
    assert backend._controller_drains
    assert not any(task.done() for task in backend._controller_drains)

    result = await backend.execute("printf later", timeout=2)
    assert result.stdout == "later"
    assert process.returncode is None

    await backend.close()
    assert process.terminated is True
    assert backend._controller is None
    assert backend._controller_drains == ()


async def test_docker_failed_cleanup_can_be_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    removals = 0

    async def run(
        argv: Any, *, input_data: bytes | None = None, timeout: float | None = None
    ) -> _ProcessResult:
        nonlocal removals
        del input_data, timeout
        arguments = tuple(argv)
        if arguments[1:3] == ("rm", "-f"):
            removals += 1
            if removals == 1:
                return _ProcessResult(1, b"", b"daemon disconnected")
        return _ProcessResult(0, b"ok", b"")

    monkeypatch.setattr(docker_backend, "_run_process", run)
    backend = DockerBackend(executable="docker")
    await backend.open(SandboxConfig(provider="docker", image="python:3.12"))
    container = backend._container_name
    with pytest.raises(SandboxError, match="cleanup failed"):
        await backend.close()
    assert backend._container_name == container
    await backend.close()
    assert removals == 2
    assert backend._container_name is None


async def test_docker_partial_open_removes_created_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    removed: list[str] = []

    async def run(
        argv: Any, *, input_data: bytes | None = None, timeout: float | None = None
    ) -> _ProcessResult:
        del input_data, timeout
        arguments = tuple(argv)
        if arguments[1] == "start":
            return _ProcessResult(1, b"", b"start failed")
        if arguments[1:3] == ("rm", "-f"):
            removed.append(arguments[3])
        return _ProcessResult(0, b"ok", b"")

    monkeypatch.setattr(docker_backend, "_run_process", run)
    backend = DockerBackend(executable="docker")
    with pytest.raises(SandboxError, match="start failed"):
        await backend.open(SandboxConfig(provider="docker", image="python:3.12"))
    assert len(removed) == 1
    assert backend._container_name is None


class _HangingProcess:
    def __init__(self) -> None:
        self.returncode: int | None = None
        self.released = asyncio.Event()
        self.killed = False

    async def communicate(self, data: bytes | None = None) -> tuple[bytes, bytes]:
        del data
        await self.released.wait()
        return b"", b""

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9
        self.released.set()


async def test_docker_subprocess_timeout_kills_and_drains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _HangingProcess()

    async def spawn(argv: Any, **kwargs: Any) -> Any:
        del argv, kwargs
        return process

    monkeypatch.setattr(docker_backend, "_spawn_process", spawn)
    with pytest.raises(TimeoutError):
        await docker_backend._run_process(("docker", "info"), timeout=0.001)
    assert process.killed is True
    assert process.returncode == -9


async def test_docker_execute_timeout_removes_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    removed: list[str] = []

    async def run(
        argv: Any, *, input_data: bytes | None = None, timeout: float | None = None
    ) -> _ProcessResult:
        del input_data, timeout
        arguments = tuple(argv)
        if arguments[1:2] == ("exec",):
            raise TimeoutError
        if arguments[1:3] == ("rm", "-f"):
            removed.append(arguments[3])
        return _ProcessResult(0, b"ok", b"")

    monkeypatch.setattr(docker_backend, "_run_process", run)
    backend = DockerBackend(executable="docker")
    await backend.open(SandboxConfig(provider="docker", image="python:3.12"))
    container = backend._container_name
    with pytest.raises(TimeoutError):
        await backend.execute("sleep forever", timeout=0.001)
    assert removed == [container]
    assert backend._container_name is None


async def test_docker_execute_cancellation_removes_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command_started = asyncio.Event()
    command_cancelled = asyncio.Event()
    removed: list[str] = []

    async def run(
        argv: Any, *, input_data: bytes | None = None, timeout: float | None = None
    ) -> _ProcessResult:
        del input_data, timeout
        arguments = tuple(argv)
        if arguments[1:2] == ("exec",):
            command_started.set()
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                command_cancelled.set()
                raise
        if arguments[1:3] == ("rm", "-f"):
            removed.append(arguments[3])
        return _ProcessResult(0, b"ok", b"")

    monkeypatch.setattr(docker_backend, "_run_process", run)
    backend = DockerBackend(executable="docker")
    await backend.open(SandboxConfig(provider="docker", image="python:3.12"))
    container = backend._container_name
    task = asyncio.create_task(backend.execute("sleep forever", timeout=60))
    await command_started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert command_cancelled.is_set()
    assert removed == [container]
    assert backend._container_name is None


class SandboxNotFoundError(Exception):
    pass


class ExecTimeoutError(Exception):
    pass


class _FakeFilesystem:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}

    async def write(self, path: str, data: bytes) -> None:
        self.files[path] = data

    async def read(self, path: str) -> bytes:
        return self.files[path]


class _ExecHandle:
    def __init__(
        self,
        events: list[Any] | None = None,
        *,
        persistent: bool = False,
    ) -> None:
        self.events = events or []
        self.persistent = persistent
        self.killed = False
        self.kill_failures = 0
        self.released = asyncio.Event()

    async def __aiter__(self):
        if self.persistent:
            await self.released.wait()
        for event in self.events:
            yield event

    async def wait(self) -> tuple[int, bool]:
        if self.persistent:
            await self.released.wait()
        return 0, True

    def kill(self) -> asyncio.Future[None]:
        # The native PyO3 SDK returns a pending Future, not a coroutine.
        loop = asyncio.get_running_loop()
        future = loop.create_future()

        def finish() -> None:
            if future.cancelled():
                return
            if self.kill_failures:
                self.kill_failures -= 1
                future.set_exception(RuntimeError("temporary kill failure"))
            else:
                self.killed = True
                self.released.set()
                future.set_result(None)

        loop.call_soon(finish)
        return future


class _FakeSandbox:
    def __init__(self) -> None:
        self.fs = _FakeFilesystem()
        self.destroy_failures = 0
        self.destroyed = False
        self.hang_commands = False
        self.exec_calls: list[tuple[str, list[str], dict[str, Any]]] = []
        self.controller: _ExecHandle | None = None
        self.last_exec_handle: _ExecHandle | None = None

    async def exec_stream(
        self, command: str, args: list[str], **kwargs: Any
    ) -> _ExecHandle:
        self.exec_calls.append((command, args, kwargs))
        if command == "controller":
            self.controller = _ExecHandle(persistent=True)
            return self.controller
        self.last_exec_handle = _ExecHandle(
            (
                []
                if self.hang_commands
                else [
                    SimpleNamespace(event_type="stdout", data=b"out\x00", code=None),
                    SimpleNamespace(event_type="stderr", data=b"err", code=None),
                    SimpleNamespace(event_type="exited", data=None, code=7),
                ]
            ),
            persistent=self.hang_commands,
        )
        return self.last_exec_handle

    async def destroy(self, *, force: bool, timeout: float | None) -> None:
        assert force is True
        assert timeout is not None
        if self.destroy_failures:
            self.destroy_failures -= 1
            raise RuntimeError("temporary destroy failure")
        self.destroyed = True


class _FakeSandboxApi:
    def __init__(self) -> None:
        self.created: _FakeSandbox | None = None
        self.create_name: str | None = None
        self.create_kwargs: dict[str, Any] = {}
        self.raise_after_create: Exception | None = None

    async def create(self, name: str, **kwargs: Any) -> _FakeSandbox:
        self.create_name = name
        self.create_kwargs = kwargs
        self.created = _FakeSandbox()
        if self.raise_after_create is not None:
            raise self.raise_after_create
        return self.created

    async def get(self, name: str) -> _FakeSandbox:
        if name != self.create_name or self.created is None or self.created.destroyed:
            raise SandboxNotFoundError(name)
        return self.created


class _FakeNetwork:
    @staticmethod
    def allow_all() -> str:
        return "network-for-package-install"


class _FakeSecurityProfile:
    RESTRICTED = "restricted-profile"


def _fake_sdk() -> Any:
    return SimpleNamespace(
        Sandbox=_FakeSandboxApi(),
        Network=_FakeNetwork,
        SecurityProfile=_FakeSecurityProfile,
        SandboxNotFoundError=SandboxNotFoundError,
        ExecTimeoutError=ExecTimeoutError,
    )


async def _open_microsandbox(
    monkeypatch: pytest.MonkeyPatch, config: SandboxConfig | None = None
) -> tuple[Any, MicrosandboxBackend]:
    sdk = _fake_sdk()
    monkeypatch.setattr(microsandbox_backend, "_check_host_support", lambda: None)
    monkeypatch.setattr(microsandbox_backend, "_load_microsandbox", lambda: sdk)
    backend = MicrosandboxBackend()
    await backend.open(
        config or SandboxConfig(provider="microsandbox", image="python:3.12-slim")
    )
    return sdk, backend


async def test_microsandbox_honors_native_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk, backend = await _open_microsandbox(
        monkeypatch,
        SandboxConfig(
            provider="microsandbox",
            image="python:3.12-slim",
            cpu_cores=3.0,
            memory_mb=1024,
            max_lifetime_seconds=600,
        ),
    )
    assert sdk.Sandbox.create_name.startswith("sqlsaber-sandbox-")
    assert sdk.Sandbox.create_kwargs == {
        "image": "python:3.12-slim",
        "network": "network-for-package-install",
        "security": "restricted-profile",
        "ephemeral": True,
        "max_duration": 600.0,
        "cpus": 3,
        "memory": 1024,
    }
    await backend.close()


async def test_microsandbox_uses_default_image_when_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk, backend = await _open_microsandbox(
        monkeypatch, SandboxConfig(provider="microsandbox")
    )
    assert sdk.Sandbox.create_kwargs["image"] == (
        "quay.io/jupyter/scipy-notebook@sha256:"
        "e6e8baae46b5e62bbc26910169082639a6fd96f90e9f6fc52e0c0389df92d35c"
    )
    await backend.close()


@pytest.mark.parametrize("cpu", [0.5, 1.5, 2.9])
async def test_microsandbox_rejects_fractional_cpu(cpu: float) -> None:
    backend = MicrosandboxBackend()
    with pytest.raises(ValueError, match="whole number"):
        await backend.open(
            SandboxConfig(provider="microsandbox", image="python:3.12", cpu_cores=cpu)
        )


async def test_microsandbox_rejects_gpu_instead_of_ignoring_it() -> None:
    backend = MicrosandboxBackend()
    with pytest.raises(ValueError, match="does not support GPU"):
        await backend.open(
            SandboxConfig(provider="microsandbox", image="python:3.12", gpu="all")
        )


async def test_microsandbox_controller_survives_exec_and_binary_transfer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk, backend = await _open_microsandbox(monkeypatch)
    sandbox = sdk.Sandbox.created
    assert sandbox is not None
    await backend.start_controller(("controller", "serve"))
    assert sandbox.controller is not None
    assert sandbox.controller.killed is False
    assert backend._controller_drain is not None
    assert not backend._controller_drain.done()
    drain = backend._controller_drain

    result = await backend.execute("exit 7", timeout=4.5)
    assert result.stdout == "out\x00"
    assert result.stderr == "err"
    assert result.exit_code == 7
    assert sandbox.exec_calls[-1] == (
        "sh",
        ["-lc", "exit 7"],
        {"timeout": 4.5},
    )
    assert sandbox.controller.killed is False

    payload = bytes(range(256)) + b"\x00\xff"
    await backend.upload(payload, "/tmp/data.bin")
    assert await backend.download("/tmp/data.bin") == payload

    await backend.close()
    assert sandbox.controller.killed is True
    assert sandbox.destroyed is True
    assert drain.done()
    assert backend._controller_handle is None
    assert backend._controller_drain is None


async def test_microsandbox_failed_destroy_can_be_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk, backend = await _open_microsandbox(monkeypatch)
    sandbox = sdk.Sandbox.created
    assert sandbox is not None
    sandbox.destroy_failures = 1
    name = backend._name
    with pytest.raises(SandboxError, match="cleanup failed"):
        await backend.close()
    assert backend._name == name
    assert backend._sandbox is sandbox
    await backend.close()
    assert sandbox.destroyed is True
    assert backend._sandbox is None


async def test_microsandbox_failed_kill_and_destroy_retain_controller_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk, backend = await _open_microsandbox(monkeypatch)
    sandbox = sdk.Sandbox.created
    assert sandbox is not None
    await backend.start_controller(("controller", "serve"))
    controller = sandbox.controller
    assert controller is not None
    controller.kill_failures = 1
    sandbox.destroy_failures = 1

    with pytest.raises(SandboxError, match="cleanup failed"):
        await backend.close()
    assert backend._controller_handle is controller
    assert backend._controller_drain is not None
    assert not backend._controller_drain.done()

    await backend.close()
    assert controller.killed is True
    assert sandbox.destroyed is True
    assert backend._controller_handle is None


async def test_microsandbox_timeout_kills_command_without_losing_controller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk, backend = await _open_microsandbox(monkeypatch)
    sandbox = sdk.Sandbox.created
    assert sandbox is not None
    await backend.start_controller(("controller", "serve"))
    sandbox.hang_commands = True

    with pytest.raises(TimeoutError):
        await backend.execute("sleep forever", timeout=0.001)
    assert sandbox.last_exec_handle is not None
    assert sandbox.last_exec_handle.killed is True
    assert sandbox.controller is not None
    assert sandbox.controller.killed is False
    await backend.close()


async def test_microsandbox_partial_create_is_cleaned_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _fake_sdk()
    sdk.Sandbox.raise_after_create = RuntimeError("create acknowledgement lost")
    monkeypatch.setattr(microsandbox_backend, "_check_host_support", lambda: None)
    monkeypatch.setattr(microsandbox_backend, "_load_microsandbox", lambda: sdk)
    backend = MicrosandboxBackend()
    with pytest.raises(SandboxError, match="create acknowledgement lost"):
        await backend.open(SandboxConfig(provider="microsandbox", image="python:3.12"))
    assert sdk.Sandbox.created is not None
    assert sdk.Sandbox.created.destroyed is True
    assert backend._name is None


@pytest.mark.skipif(
    sys.platform != "linux" or os.access("/dev/kvm", os.R_OK | os.W_OK),
    reason="test requires Linux without accessible KVM",
)
async def test_microsandbox_preflight_reports_inaccessible_kvm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_sdk_load() -> Any:
        raise AssertionError("Microsandbox SDK loaded before the host preflight")

    monkeypatch.setattr(microsandbox_backend, "_load_microsandbox", unexpected_sdk_load)
    backend = MicrosandboxBackend()
    with pytest.raises(SandboxError, match="readable and writable /dev/kvm"):
        await backend.open(SandboxConfig(provider="microsandbox"))


def test_installed_microsandbox_native_api_contract() -> None:
    sdk = pytest.importorskip("microsandbox")
    assert sdk.version().startswith("0.6.")
    assert isinstance(sdk.Network.allow_all(), sdk.Network)
    assert str(sdk.SecurityProfile.RESTRICTED) == "restricted"

    create = inspect.signature(sdk.Sandbox.create)
    assert "name" in create.parameters
    assert any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in create.parameters.values()
    )
    assert {"cmd", "args", "timeout"} <= set(
        inspect.signature(sdk.Sandbox.exec_stream).parameters
    )
    assert {"force", "timeout"} <= set(
        inspect.signature(sdk.Sandbox.destroy).parameters
    )
    assert list(inspect.signature(sdk.SandboxFsOps.write).parameters)[-2:] == [
        "path",
        "data",
    ]
    assert list(inspect.signature(sdk.SandboxFsOps.read).parameters)[-1:] == ["path"]
