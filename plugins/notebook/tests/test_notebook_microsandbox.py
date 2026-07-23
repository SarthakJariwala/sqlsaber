from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest

from sqlsaber_notebook.execution import (
    ExecutionLimits,
    NotebookBackendUnavailable,
    NotebookExecutionTimeout,
    NotebookImageError,
    NotebookInput,
    NotebookLimitExceeded,
)
from sqlsaber_notebook.execution.base import NotebookInfrastructureError
from sqlsaber_notebook.execution import microsandbox as microsandbox_backend


class SandboxNotFoundError(Exception):
    pass


class ImagePullFailedError(Exception):
    pass


class ExecTimeoutError(Exception):
    pass


class IoError(Exception):
    pass


class FakeOutput:
    def __init__(
        self,
        *,
        success: bool = True,
        stdout: bytes = b"",
        stderr: bytes = b"",
    ) -> None:
        self.success = success
        self.exit_code = 0 if success else 1
        self.stdout_bytes = stdout
        self.stderr_bytes = stderr
        self.stdout_text = stdout.decode()
        self.stderr_text = stderr.decode()


class FakeEvent:
    def __init__(
        self, event_type: str, data: bytes | None = None, code: int | None = None
    ) -> None:
        self.event_type = event_type
        self.data = data
        self.code = code


class FakeExecHandle:
    def __init__(self, *, hang: bool = False, output_size: int = 0) -> None:
        self.hang = hang
        self.output_size = output_size
        self.killed = False

    async def __aiter__(self):
        if self.hang:
            await asyncio.sleep(60)
        if self.output_size:
            yield FakeEvent("stdout", b"x" * self.output_size)
        yield FakeEvent("exited", code=0)

    async def wait(self) -> tuple[int, bool]:
        return 0, True

    async def kill(self) -> None:
        self.killed = True


class FakeReadStream:
    def __init__(self, data: bytes) -> None:
        self.data = data

    async def __aiter__(self):
        for offset in range(0, len(self.data), 7):
            yield self.data[offset : offset + 7]


class FakeWriteStream:
    def __init__(self, filesystem: FakeFilesystem, path: str) -> None:
        self.filesystem = filesystem
        self.path = path
        self.data = bytearray()

    async def write(self, data: bytes) -> None:
        self.data.extend(data)

    async def __aenter__(self) -> FakeWriteStream:
        return self

    async def __aexit__(self, *args: Any) -> bool:
        self.filesystem.files[self.path] = bytes(self.data)
        return False


class FakeFilesystem:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}

    async def write_stream(self, path: str) -> FakeWriteStream:
        return FakeWriteStream(self, path)

    async def read_stream(self, path: str) -> FakeReadStream:
        return FakeReadStream(self.files[path])

    async def stat(self, path: str) -> Any:
        data = self.files[path]
        return SimpleNamespace(kind="file", size=len(data), mode=0o444)

    async def remove(self, path: str) -> None:
        if path not in self.files:
            raise FileNotFoundError(path)
        del self.files[path]


class FakeHandle:
    def __init__(self, api: FakeSandboxApi, sandbox: FakeSandbox) -> None:
        self.api = api
        self.sandbox = sandbox
        self.status = sandbox.status

    async def kill(self, timeout: float | None = None) -> None:
        del timeout
        self.sandbox.status = "stopped"
        self.status = "stopped"

    async def refresh(self) -> FakeHandle:
        self.status = self.sandbox.status
        return self

    async def remove(self) -> None:
        if self.sandbox.status == "running":
            raise RuntimeError("still running")
        self.api.removed = True


class FakeSandbox:
    def __init__(self, api: FakeSandboxApi) -> None:
        self.api = api
        self.fs = FakeFilesystem()
        self.commands: list[tuple[str, tuple[str, ...], dict[str, Any]]] = []
        self.status = "running"
        self.raise_after_kill = False
        self.hang = False
        self.output_size = 0
        self.last_exec_handle: FakeExecHandle | None = None
        self.run_count = 0
        self.artifact_path = "nested/result.txt"

    async def exec(
        self,
        command: str,
        args: list[str],
        **kwargs: Any,
    ) -> FakeOutput:
        self.commands.append((command, tuple(args), kwargs))
        if command == "rm" and microsandbox_backend._RUN_PATH in args:
            prefix = f"{microsandbox_backend._RUN_PATH}/"
            self.fs.files = {
                path: data
                for path, data in self.fs.files.items()
                if not path.startswith(prefix)
            }
        elif command == "rm" and args[:1] == ["-f"]:
            self.fs.files.pop(args[1], None)
        if (
            command == microsandbox_backend._PYTHON
            and args[1] == microsandbox_backend._INVENTORY_SCRIPT
        ):
            inventory = {
                "files": {
                    "notebook.ipynb": len(
                        self.fs.files[microsandbox_backend._NOTEBOOK_PATH]
                    ),
                    self.artifact_path: len(
                        self.fs.files[
                            f"{microsandbox_backend._RUN_PATH}/{self.artifact_path}"
                        ]
                    ),
                }
            }
            self.fs.files[microsandbox_backend._INVENTORY_PATH] = json.dumps(
                inventory
            ).encode()
        return FakeOutput(stdout=b"ok\n")

    async def exec_stream(
        self,
        command: str,
        args: list[str],
        **kwargs: Any,
    ) -> FakeExecHandle:
        self.commands.append((command, tuple(args), kwargs))
        self.run_count += 1
        self.artifact_path = (
            "nested/result.txt"
            if self.run_count == 1
            else f"nested/result-{self.run_count}.txt"
        )
        self.fs.files[f"{microsandbox_backend._RUN_PATH}/{self.artifact_path}"] = (
            b"result"
        )
        self.last_exec_handle = FakeExecHandle(
            hang=self.hang, output_size=self.output_size
        )
        return self.last_exec_handle

    async def kill(self, timeout: float | None = None) -> None:
        del timeout
        self.status = "stopped"
        if self.raise_after_kill:
            raise IoError("No child processes")


class FakeSandboxApi:
    def __init__(self) -> None:
        self.created: FakeSandbox | None = None
        self.create_name = ""
        self.create_kwargs: dict[str, Any] = {}
        self.removed = False

    async def create(self, name: str, **kwargs: Any) -> FakeSandbox:
        self.create_name = name
        self.create_kwargs = kwargs
        self.created = FakeSandbox(self)
        return self.created

    async def get(self, name: str) -> FakeHandle:
        assert name == self.create_name
        if self.created is None or self.removed:
            raise SandboxNotFoundError(name)
        return FakeHandle(self, self.created)


class FakeNetwork:
    @staticmethod
    def none() -> str:
        return "network-none"


class FakeRlimit:
    @staticmethod
    def nproc(limit: int) -> tuple[str, int]:
        return "nproc", limit


def fake_sdk() -> Any:
    sandbox_api = FakeSandboxApi()
    return SimpleNamespace(
        Sandbox=sandbox_api,
        Network=FakeNetwork,
        Rlimit=FakeRlimit,
        SandboxNotFoundError=SandboxNotFoundError,
        ImagePullFailedError=ImagePullFailedError,
        ExecTimeoutError=ExecTimeoutError,
        IoError=IoError,
    )


def notebook_bytes() -> bytes:
    return json.dumps(
        {
            "cells": [],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
    ).encode()


async def open_environment(
    monkeypatch: pytest.MonkeyPatch,
    *,
    limits: ExecutionLimits | None = None,
):
    sdk = fake_sdk()
    monkeypatch.setattr(microsandbox_backend, "_load_microsandbox", lambda: sdk)
    environment = await microsandbox_backend.MicrosandboxNotebookBackend().open(
        [NotebookInput("data.json", b"{}")],
        image="registry/image@sha256:digest",
        limits=limits or ExecutionLimits(),
    )
    return sdk, environment


def test_missing_sdk_reports_optional_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str) -> Any:
        assert name == "microsandbox"
        raise ImportError(name)

    monkeypatch.setattr(microsandbox_backend.importlib, "import_module", missing)
    with pytest.raises(NotebookBackendUnavailable, match="microsandbox.*extra"):
        microsandbox_backend._load_microsandbox()


async def test_registry_failure_is_normalized_as_image_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = fake_sdk()

    async def fail_create(name: str, **kwargs: Any) -> FakeSandbox:
        del name, kwargs
        raise RuntimeError("image error: registry error: Not authorized")

    sdk.Sandbox.create = fail_create
    monkeypatch.setattr(microsandbox_backend, "_load_microsandbox", lambda: sdk)
    with pytest.raises(NotebookImageError, match="Could not prepare") as raised:
        await microsandbox_backend.MicrosandboxNotebookBackend().open(
            [], image="private/image", limits=ExecutionLimits()
        )
    assert raised.value.phase == "image-prepare"


async def test_open_uses_restricted_networkless_microvm_and_immutable_staging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk, environment = await open_environment(monkeypatch)
    sandbox = sdk.Sandbox.created
    assert sandbox is not None
    assert sdk.Sandbox.create_name.startswith("sqlsaber-notebook-")
    assert sdk.Sandbox.create_kwargs == {
        "image": "registry/image@sha256:digest",
        "cpus": 4,
        "memory": 8192,
        "network": "network-none",
        "security": "restricted",
        "ephemeral": True,
        "max_duration": 86_400.0,
    }
    assert sandbox.fs.files[f"{microsandbox_backend._INPUTS_PATH}/data.json"] == b"{}"
    assert any(
        command == "chmod" and "a-w" in args for command, args, _ in sandbox.commands
    )
    assert any(
        command == "jupyter"
        and args == ("nbconvert", "--version")
        and kwargs["user"] == "1000"
        for command, args, kwargs in sandbox.commands
    )
    await environment.close()
    await environment.close()
    assert sdk.Sandbox.removed is True


async def test_execute_streams_as_uid_1000_with_pid_limit_and_lazy_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, environment = await open_environment(monkeypatch)
    result = await environment.execute(
        notebook_bytes(), cell_timeout=30, command_timeout=60
    )
    assert [artifact.path for artifact in result.artifacts] == ["nested/result.txt"]
    assert await environment.read_artifact(result.artifacts[0]) == b"result"

    stream_commands = [
        item
        for item in environment.sandbox.commands
        if item[0] == "jupyter" and "--execute" in item[1]
    ]
    assert len(stream_commands) == 1
    _, args, kwargs = stream_commands[0]
    assert "--ExecutePreprocessor.timeout=30" in args
    assert kwargs["cwd"] == microsandbox_backend._RUN_PATH
    assert kwargs["user"] == "1000"
    assert kwargs["rlimits"] == [("nproc", 256)]
    await environment.close()


async def test_execution_timeout_kills_process_and_removes_sandbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk, environment = await open_environment(monkeypatch)
    environment.sandbox.hang = True
    with pytest.raises(NotebookExecutionTimeout, match="timed out"):
        await environment.execute(
            notebook_bytes(),
            cell_timeout=1,
            command_timeout=0.01,  # type: ignore[arg-type]
        )
    assert environment.sandbox.last_exec_handle is not None
    assert environment.sandbox.last_exec_handle.killed is True
    assert sdk.Sandbox.removed is True
    with pytest.raises(NotebookInfrastructureError, match="closed"):
        await environment.list_workspace()


async def test_execution_cancellation_kills_process_and_removes_sandbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk, environment = await open_environment(monkeypatch)
    environment.sandbox.hang = True
    task = asyncio.create_task(
        environment.execute(notebook_bytes(), cell_timeout=1, command_timeout=None)
    )
    while environment.sandbox.last_exec_handle is None:
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert environment.sandbox.last_exec_handle.killed is True
    assert sdk.Sandbox.removed is True


async def test_streamed_execution_logs_are_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, environment = await open_environment(monkeypatch)
    environment.sandbox.output_size = 1_000_000
    result = await environment.execute(
        notebook_bytes(), cell_timeout=1, command_timeout=2
    )
    assert len(result.stdout) <= environment.limits.max_log_chars
    assert "backend log truncated" in result.stdout
    await environment.close()


async def test_cleanup_verifies_state_when_kill_raises_no_child_processes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk, environment = await open_environment(monkeypatch)
    environment.sandbox.raise_after_kill = True
    await environment.close()
    assert sdk.Sandbox.removed is True


async def test_stale_artifact_is_rejected_after_next_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, environment = await open_environment(monkeypatch)
    first = await environment.execute(
        notebook_bytes(), cell_timeout=1, command_timeout=2
    )
    await environment.execute(notebook_bytes(), cell_timeout=1, command_timeout=2)
    with pytest.raises(NotebookInfrastructureError, match="Unknown artifact"):
        await environment.read_artifact(first.artifacts[0])
    await environment.close()


@pytest.mark.parametrize(("value", "expected"), [(4.0, 4), (2.9, 2), (1.0, 1)])
def test_cpu_mapping_is_conservative(value: float, expected: int) -> None:
    assert microsandbox_backend._microsandbox_cpus(value) == expected


@pytest.mark.parametrize("value", [0.9, 0.0, float("inf"), float("nan")])
def test_cpu_mapping_rejects_unsupported_values(value: float) -> None:
    with pytest.raises(NotebookLimitExceeded):
        microsandbox_backend._microsandbox_cpus(value)
