from __future__ import annotations

import asyncio
import base64
import json
import shlex
from pathlib import Path
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
from sqlsaber_notebook.execution import daytona as daytona_backend
from sqlsaber_notebook.execution.base import NotebookInfrastructureError


class DaytonaError(Exception):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class DaytonaNotFoundError(DaytonaError):
    pass


class DaytonaRateLimitError(DaytonaError):
    pass


class DaytonaTimeoutError(DaytonaError):
    pass


class FakeImage:
    def __init__(self, parent: str) -> None:
        self.parent = parent
        self.commands: list[str] = []

    def dockerfile_commands(self, commands: list[str]) -> FakeImage:
        self.commands = list(commands)
        return self


class FakeImageFactory:
    created: list[FakeImage] = []

    @classmethod
    def base(cls, parent: str) -> FakeImage:
        image = FakeImage(parent)
        cls.created.append(image)
        return image


class FakeResources:
    def __init__(self, *, cpu: int, memory: int) -> None:
        self.cpu = cpu
        self.memory = memory


class FakeCreateParams:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


class FakeFilesystem:
    def __init__(self, sandbox: FakeSandbox) -> None:
        self.sandbox = sandbox
        self.files: dict[str, bytes] = {}
        self.identities: dict[str, tuple[int, int, int]] = {}
        self.next_inode = 100
        self.download_paths: list[str] = []
        self.short_download = False

    def put(self, path: str, data: bytes) -> None:
        self.files[path] = bytes(data)
        prior = self.identities.get(path)
        inode = prior[1] if prior else self.next_inode
        if prior is None:
            self.next_inode += 1
        self.identities[path] = (1, inode, len(data))

    async def upload_file(
        self,
        source: bytes,
        destination: str,
        timeout: int,
    ) -> None:
        assert timeout > 0
        self.put(destination, source)

    async def download_file(self, *args: str) -> None:
        remote, local, _timeout = args
        self.download_paths.append(local)
        data = self.files[remote]
        if self.short_download:
            data = data[:-1]
        Path(local).write_bytes(data)


class FakeProcess:
    def __init__(self, sandbox: FakeSandbox) -> None:
        self.sandbox = sandbox
        self.calls: list[tuple[str, tuple[str, ...], str | None, int | None]] = []
        self.hang = False
        self.fail_preflight = False
        self.execution_count = 0

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> Any:
        del env
        argv = tuple(shlex.split(command))
        self.calls.append((command, argv, cwd, timeout))
        if self.hang and daytona_backend._LOG_WRAPPER in argv:
            await asyncio.sleep(60)
        if self.fail_preflight and daytona_backend._ROOT_PREFLIGHT_SCRIPT in argv:
            return SimpleNamespace(exit_code=1, result="missing root")

        files = self.sandbox.fs.files
        if argv[:2] == ("rm", "-rf"):
            prefix = f"{argv[-1].rstrip('/')}/"
            for path in tuple(files):
                if path == argv[-1] or path.startswith(prefix):
                    files.pop(path, None)
                    self.sandbox.fs.identities.pop(path, None)
        elif argv[:2] == ("rm", "-f"):
            for path in argv[2:]:
                files.pop(path, None)
                self.sandbox.fs.identities.pop(path, None)
        elif daytona_backend._LOG_WRAPPER in argv:
            self.execution_count += 1
            wrapper_index = argv.index(daytona_backend._LOG_WRAPPER)
            stdout_path = argv[wrapper_index + 2]
            stderr_path = argv[wrapper_index + 3]
            self.sandbox.fs.put(stdout_path, b"executed\n")
            self.sandbox.fs.put(stderr_path, b"")
            artifact = f"{cwd}/nested/result-{self.execution_count}.txt"
            self.sandbox.fs.put(artifact, f"run-{self.execution_count}".encode())
        elif daytona_backend._INVENTORY_SCRIPT in argv:
            script_index = argv.index(daytona_backend._INVENTORY_SCRIPT)
            run_path = argv[script_index + 1]
            output_path = argv[script_index + 2]
            prefix = f"{run_path}/"
            entries: dict[str, dict[str, int]] = {}
            for path, data in sorted(files.items()):
                if path.startswith(prefix):
                    relative = path[len(prefix) :]
                    device, inode, size = self.sandbox.fs.identities[path]
                    entries[relative] = {
                        "size": size,
                        "device": device,
                        "inode": inode,
                    }
            self.sandbox.fs.put(
                output_path,
                json.dumps({"files": entries}, separators=(",", ":")).encode(),
            )
        elif daytona_backend._METADATA_SCRIPT in argv:
            script_index = argv.index(daytona_backend._METADATA_SCRIPT)
            path = argv[script_index + 1]
            device, inode, size = self.sandbox.fs.identities[path]
            return SimpleNamespace(
                exit_code=0,
                result=json.dumps(
                    {
                        "regular": True,
                        "size": size,
                        "device": device,
                        "inode": inode,
                    },
                    separators=(",", ":"),
                ),
            )
        elif daytona_backend._LOG_READER_SCRIPT in argv:
            script_index = argv.index(daytona_backend._LOG_READER_SCRIPT)
            stdout_path = argv[script_index + 1]
            stderr_path = argv[script_index + 2]
            return SimpleNamespace(
                exit_code=0,
                result=json.dumps(
                    {
                        "stdout": base64.b64encode(
                            files.get(stdout_path, b"")
                        ).decode(),
                        "stderr": base64.b64encode(
                            files.get(stderr_path, b"")
                        ).decode(),
                    },
                    separators=(",", ":"),
                ),
            )
        return SimpleNamespace(exit_code=0, result="ok")


class FakeSandbox:
    def __init__(self, client: FakeClient, name: str) -> None:
        self.client = client
        self.name = name
        self.fs = FakeFilesystem(self)
        self.process = FakeProcess(self)
        self.delete_calls = 0

    async def get_work_dir(self) -> str:
        return "/home/jovyan"

    async def delete(self, timeout: float | None = None) -> None:
        assert timeout == 60
        self.delete_calls += 1
        self.client.deleted = True


class FakeClient:
    def __init__(self, sdk: Any) -> None:
        self.sdk = sdk
        self.create_params: FakeCreateParams | None = None
        self.create_timeout: float | None = None
        self.sandbox: FakeSandbox | None = None
        self.deleted = False
        self.closed = False
        self.get_calls = 0
        self.create_error: Exception | None = None
        sdk.clients.append(self)

    async def create(self, params: FakeCreateParams, *, timeout: float) -> FakeSandbox:
        self.create_params = params
        self.create_timeout = timeout
        name = params.name
        self.sandbox = FakeSandbox(self, name)
        if self.sdk.create_error is not None:
            raise self.sdk.create_error
        return self.sandbox

    async def get(self, name: str) -> FakeSandbox:
        self.get_calls += 1
        if self.sandbox is None or self.deleted:
            raise DaytonaNotFoundError(name)
        assert name == self.sandbox.name
        return self.sandbox

    async def close(self) -> None:
        self.closed = True


def fake_sdk() -> Any:
    FakeImageFactory.created = []
    sdk = SimpleNamespace(
        clients=[],
        create_error=None,
        CreateSandboxFromImageParams=FakeCreateParams,
        CreateSandboxFromSnapshotParams=FakeCreateParams,
        Image=FakeImageFactory,
        Resources=FakeResources,
        DaytonaError=DaytonaError,
        DaytonaNotFoundError=DaytonaNotFoundError,
        DaytonaRateLimitError=DaytonaRateLimitError,
        DaytonaTimeoutError=DaytonaTimeoutError,
    )
    sdk.AsyncDaytona = lambda: FakeClient(sdk)
    return sdk


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
    monkeypatch.setattr(daytona_backend, "_load_daytona", lambda: sdk)
    environment = await daytona_backend.DaytonaNotebookBackend().open(
        [NotebookInput("data file;$x.json", b"{}")],
        image="registry/image@sha256:digest",
        limits=limits or ExecutionLimits(),
    )
    return sdk, environment


def test_missing_sdk_reports_optional_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str) -> Any:
        assert name == "daytona"
        raise ImportError(name)

    monkeypatch.setattr(daytona_backend.importlib, "import_module", missing)
    with pytest.raises(NotebookBackendUnavailable, match="daytona.*extra"):
        daytona_backend._load_daytona()


def test_incompatible_sdk_version_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        daytona_backend.importlib, "import_module", lambda name: object()
    )
    monkeypatch.setattr(daytona_backend, "version", lambda name: "0.168.0")
    with pytest.raises(NotebookBackendUnavailable, match="requires daytona==0.143.0"):
        daytona_backend._load_daytona()


def test_incompatible_sdk_symbols_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        daytona_backend.importlib, "import_module", lambda name: object()
    )
    monkeypatch.setattr(daytona_backend, "version", lambda name: "0.143.0")
    with pytest.raises(NotebookBackendUnavailable, match="missing: AsyncDaytona"):
        daytona_backend._load_daytona()


async def test_open_uses_root_wrapper_blocked_network_and_immutable_staging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk, environment = await open_environment(monkeypatch)
    client = sdk.clients[0]
    params = client.create_params
    assert params is not None
    image = FakeImageFactory.created[0]
    assert image.parent == "registry/image@sha256:digest"
    assert image.commands == ["USER root", "WORKDIR /home/jovyan"]
    assert params.image is image
    assert params.name.startswith("sqlsaber-notebook-")
    assert params.labels == {"application": "sqlsaber", "purpose": "notebook"}
    assert params.os_user == "root"
    assert params.network_block_all is True
    assert params.ephemeral is True
    assert params.auto_stop_interval == 1440
    assert params.resources.cpu == 4
    assert params.resources.memory == 8
    assert client.create_timeout == 600
    assert environment.remote_root.startswith("/tmp/sqlsaber-notebook/")
    assert (
        environment.sandbox.fs.files[f"{environment.inputs_path}/data file;$x.json"]
        == b"{}"
    )
    commands = [call[1] for call in environment.sandbox.process.calls]
    assert any(daytona_backend._SECURE_PARENT_SCRIPT in command for command in commands)
    assert any(command[:3] == ("chown", "-R", "root:root") for command in commands)
    assert any(command[:3] == ("chmod", "-R", "a-w") for command in commands)
    assert any(
        command[:5] == (daytona_backend._RUNUSER, "-u", "jovyan", "--", "jupyter")
        for command in commands
    )
    await environment.close()


async def test_open_can_use_a_named_daytona_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = fake_sdk()
    monkeypatch.setattr(daytona_backend, "_load_daytona", lambda: sdk)

    environment = await daytona_backend.DaytonaNotebookBackend().open(
        [],
        image="unused",
        snapshot="analytics-ready",
        limits=ExecutionLimits(),
    )

    params = sdk.clients[0].create_params
    assert params is not None
    assert params.snapshot == "analytics-ready"
    assert params.os_user == "root"
    assert params.network_block_all is True
    assert params.ephemeral is True
    assert FakeImageFactory.created == []
    await environment.close()


async def test_execute_has_clean_runs_bounded_logs_and_lazy_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, environment = await open_environment(monkeypatch)
    first = await environment.execute(
        notebook_bytes(), cell_timeout=30, command_timeout=60
    )
    assert first.stdout == "executed\n"
    assert [artifact.path for artifact in first.artifacts] == ["nested/result-1.txt"]
    assert await environment.read_artifact(first.artifacts[0]) == b"run-1"
    assert all(
        not Path(path).exists() for path in environment.sandbox.fs.download_paths
    )

    second = await environment.execute(
        notebook_bytes(), cell_timeout=45, command_timeout=90
    )
    assert [artifact.path for artifact in second.artifacts] == ["nested/result-2.txt"]
    with pytest.raises(NotebookInfrastructureError, match="Unknown artifact"):
        await environment.read_artifact(first.artifacts[0])

    execution_calls = [
        call
        for call in environment.sandbox.process.calls
        if daytona_backend._LOG_WRAPPER in call[1]
    ]
    assert len(execution_calls) == 2
    assert execution_calls[0][2] == environment.run_path
    assert execution_calls[0][3] == 60
    assert "--ExecutePreprocessor.timeout=30" in execution_calls[0][1]
    await environment.close()


async def test_short_legacy_download_is_rejected_and_temporary_file_removed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, environment = await open_environment(monkeypatch)
    environment.sandbox.fs.short_download = True
    with pytest.raises(NotebookInfrastructureError, match="changed during download"):
        await environment.execute(notebook_bytes(), cell_timeout=10, command_timeout=20)
    assert all(
        not Path(path).exists() for path in environment.sandbox.fs.download_paths
    )
    await environment.close()


async def test_preflight_failure_deletes_sandbox_and_closes_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = fake_sdk()
    original_client = FakeClient(sdk)

    def client_factory() -> FakeClient:
        return original_client

    sdk.AsyncDaytona = client_factory
    monkeypatch.setattr(daytona_backend, "_load_daytona", lambda: sdk)

    original_create = original_client.create

    async def create(*args: Any, **kwargs: Any) -> FakeSandbox:
        sandbox = await original_create(*args, **kwargs)
        sandbox.process.fail_preflight = True
        return sandbox

    original_client.create = create
    with pytest.raises(NotebookImageError, match="root Daytona control"):
        await daytona_backend.DaytonaNotebookBackend().open(
            [], image="image", limits=ExecutionLimits()
        )
    assert original_client.sandbox is not None
    assert original_client.sandbox.delete_calls == 1
    assert original_client.closed is True


async def test_execution_cancellation_deletes_sandbox_and_closes_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk, environment = await open_environment(monkeypatch)
    environment.sandbox.process.hang = True
    task = asyncio.create_task(
        environment.execute(notebook_bytes(), cell_timeout=1, command_timeout=None)
    )
    while not any(
        daytona_backend._LOG_WRAPPER in call[1]
        for call in environment.sandbox.process.calls
    ):
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert environment.sandbox.delete_calls == 1
    assert sdk.clients[0].closed is True
    with pytest.raises(NotebookInfrastructureError, match="closed"):
        await environment.list_workspace()


@pytest.mark.parametrize(
    "method_name",
    ["_read_remote_logs", "_stop_run_processes", "_freeze_run"],
)
async def test_post_execution_security_failure_deletes_sandbox(
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
) -> None:
    sdk, environment = await open_environment(monkeypatch)

    async def fail() -> None:
        raise NotebookInfrastructureError(
            "post-execution security failed",
            backend="daytona",
            phase="notebook-execution",
        )

    monkeypatch.setattr(environment, method_name, fail)
    with pytest.raises(
        NotebookInfrastructureError,
        match="post-execution security failed",
    ):
        await environment.execute(
            notebook_bytes(),
            cell_timeout=1,
            command_timeout=2,
        )
    assert environment.sandbox.delete_calls == 1
    assert sdk.clients[0].closed is True
    with pytest.raises(NotebookInfrastructureError, match="closed"):
        await environment.list_workspace()


async def test_sdk_exec_failure_preserves_operation_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk, environment = await open_environment(monkeypatch)
    original_exec = environment.sandbox.process.exec

    async def fail_log_read(
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> Any:
        if daytona_backend._LOG_READER_SCRIPT in shlex.split(command):
            raise DaytonaError("toolbox failed")
        return await original_exec(command, cwd=cwd, env=env, timeout=timeout)

    monkeypatch.setattr(environment.sandbox.process, "exec", fail_log_read)
    with pytest.raises(NotebookInfrastructureError) as raised:
        await environment.execute(
            notebook_bytes(),
            cell_timeout=1,
            command_timeout=2,
        )
    assert raised.value.phase == "notebook-execution"
    assert "toolbox failed" in raised.value.diagnostics
    assert environment.sandbox.delete_calls == 1
    assert sdk.clients[0].closed is True


async def test_sdk_transfer_failure_preserves_operation_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, environment = await open_environment(monkeypatch)
    original_download = environment.sandbox.fs.download_file

    async def fail_inventory_download(*args: str) -> None:
        if args[0] == environment.inventory_path:
            raise DaytonaError("transfer failed")
        await original_download(*args)

    monkeypatch.setattr(
        environment.sandbox.fs,
        "download_file",
        fail_inventory_download,
    )
    with pytest.raises(NotebookInfrastructureError) as raised:
        await environment.execute(
            notebook_bytes(),
            cell_timeout=1,
            command_timeout=2,
        )
    assert raised.value.phase == "artifact-inventory"
    assert "transfer failed" in raised.value.diagnostics
    assert environment.sandbox.delete_calls == 0
    await environment.close()


async def test_sdk_execution_timeout_deletes_sandbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk, environment = await open_environment(monkeypatch)

    async def timeout(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise DaytonaTimeoutError("command timeout")

    environment.sandbox.process.exec = timeout
    with pytest.raises(NotebookExecutionTimeout):
        await environment.execute(notebook_bytes(), cell_timeout=1, command_timeout=2)
    assert environment.sandbox.delete_calls == 1
    assert sdk.clients[0].closed is True


async def test_close_is_idempotent_and_polls_until_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk, environment = await open_environment(monkeypatch)
    await environment.close()
    await environment.close()
    client = sdk.clients[0]
    assert environment.sandbox.delete_calls == 1
    assert client.get_calls == 1
    assert client.closed is True


async def test_deletion_polling_times_out_when_sandbox_remains() -> None:
    sdk = fake_sdk()
    client = FakeClient(sdk)
    client.sandbox = FakeSandbox(client, "still-there")
    with pytest.raises(TimeoutError, match="still exists"):
        await daytona_backend._wait_until_deleted(
            client,
            "still-there",
            sdk,
            timeout=0.01,
            poll_interval=0,
        )


async def test_client_closes_when_sandbox_deletion_fails() -> None:
    sdk = fake_sdk()
    client = FakeClient(sdk)
    sandbox = FakeSandbox(client, "delete-failure")

    async def fail_delete(timeout: float | None = None) -> None:
        del timeout
        raise DaytonaError("delete failed")

    sandbox.delete = fail_delete
    with pytest.raises(DaytonaError, match="delete failed"):
        await daytona_backend._delete_resources(
            sdk,
            client,
            sandbox,
            sandbox.name,
        )
    assert client.closed is True


async def test_already_deleted_sandbox_is_successful_cleanup() -> None:
    sdk = fake_sdk()
    client = FakeClient(sdk)
    sandbox = FakeSandbox(client, "already-deleted")

    async def already_deleted(timeout: float | None = None) -> None:
        del timeout
        raise DaytonaNotFoundError("already deleted")

    sandbox.delete = already_deleted
    await daytona_backend._delete_resources(
        sdk,
        client,
        sandbox,
        sandbox.name,
    )
    assert client.get_calls == 0
    assert client.closed is True


async def test_create_timeout_recovers_named_orphan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = fake_sdk()
    sdk.create_error = DaytonaTimeoutError("create timed out")
    monkeypatch.setattr(daytona_backend, "_load_daytona", lambda: sdk)
    with pytest.raises(NotebookExecutionTimeout, match="image preparation"):
        await daytona_backend.DaytonaNotebookBackend().open(
            [], image="image", limits=ExecutionLimits()
        )
    client = sdk.clients[0]
    assert client.sandbox is not None
    assert client.sandbox.delete_calls == 1
    assert client.closed is True


async def test_safe_exec_preserves_shell_metacharacters_as_single_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, environment = await open_environment(monkeypatch)
    values = ("space value", "quote'value", "$HOME; touch nope", "line\nvalue")
    await environment._exec("test", "printf", *values, timeout=1)
    parsed = environment.sandbox.process.calls[-1][1]
    assert parsed == ("printf", *values)
    await environment.close()


@pytest.mark.parametrize(("value", "expected"), [(4.0, 4), (2.9, 2), (1.0, 1)])
def test_cpu_mapping_is_conservative(value: float, expected: int) -> None:
    assert daytona_backend._daytona_cpu(value) == expected


@pytest.mark.parametrize("value", [0.9, 0.0, float("inf"), float("nan")])
def test_cpu_mapping_rejects_unsupported_values(value: float) -> None:
    with pytest.raises(NotebookLimitExceeded):
        daytona_backend._daytona_cpu(value)


@pytest.mark.parametrize(("value", "expected"), [(8192, 8), (2047, 1), (1024, 1)])
def test_memory_mapping_is_conservative(value: int, expected: int) -> None:
    assert daytona_backend._daytona_memory_gib(value) == expected


def test_memory_mapping_rejects_less_than_one_gib() -> None:
    with pytest.raises(NotebookLimitExceeded):
        daytona_backend._daytona_memory_gib(1023)
