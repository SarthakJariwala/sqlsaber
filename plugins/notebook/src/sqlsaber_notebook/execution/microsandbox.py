"""Local microVM notebook execution through the optional Microsandbox SDK."""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import json
import math
import uuid
from collections.abc import Sequence
from typing import Any

from ._files import build_artifact_inventory, validate_notebook_bytes
from .base import (
    ArtifactInfo,
    ExecutionLimits,
    NotebookBackend,
    NotebookBackendUnavailable,
    NotebookEnvironment,
    NotebookExecutionError,
    NotebookExecutionResult,
    NotebookExecutionTimeout,
    NotebookImageError,
    NotebookInfrastructureError,
    NotebookInput,
    NotebookLimitExceeded,
    bound_log,
    validate_artifact_path,
    validate_inputs,
)

_REMOTE_ROOT = "/work"
_INPUTS_PATH = f"{_REMOTE_ROOT}/inputs"
_RUN_PATH = f"{_REMOTE_ROOT}/run"
_CONTROL_PATH = f"{_REMOTE_ROOT}/control"
_NOTEBOOK_PATH = f"{_RUN_PATH}/notebook.ipynb"
_INVENTORY_PATH = f"{_CONTROL_PATH}/inventory.json"
_RUN_USER = "1000"
_RUN_OWNER = "1000:100"
_PYTHON = "/opt/conda/bin/python"
_PLATFORM_MAX_DURATION_SECONDS = 24 * 60 * 60
_INVENTORY_MAX_BYTES = 128 * 1024
_STREAM_CHUNK_BYTES = 1024 * 1024

_PREFLIGHT_SCRIPT = r"""
import os
import pathlib
import sys

input_path = pathlib.Path(sys.argv[1])
run_path = pathlib.Path(sys.argv[2])
if os.getuid() != 1000:
    raise SystemExit("notebook process did not run as uid 1000")
input_path.read_bytes()
probe = run_path / "preflight-write"
probe.write_bytes(b"ok")
probe.unlink()
failed = []
for operation in (
    lambda: input_path.write_bytes(b"changed"),
    lambda: input_path.chmod(0o600),
    lambda: input_path.rename(run_path / "moved-input"),
    lambda: input_path.unlink(),
    lambda: (input_path.parent / "replacement").write_bytes(b"replacement"),
):
    try:
        operation()
    except OSError:
        continue
    failed.append("mutation unexpectedly succeeded")
if failed:
    raise SystemExit(failed[0])
""".strip()

_STOP_USER_PROCESSES_SCRIPT = r"""
import os
import signal
import time

for _ in range(3):
    found = False
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        try:
            if os.stat(entry.path).st_uid == 1000:
                os.kill(int(entry.name), signal.SIGKILL)
                found = True
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    if not found:
        break
    time.sleep(0.05)
""".strip()

_INVENTORY_SCRIPT = r"""
import json
import os
import stat
import sys

root = sys.argv[1]
output = sys.argv[2]
max_entries = int(sys.argv[3]) + 1
max_file = int(sys.argv[4])
max_total = int(sys.argv[5])
max_notebook = int(sys.argv[6])
max_directories = 256
max_depth = 32
max_path = 1024
entries = {}
total = 0
directories = 0
try:
    for current, dirs, files in os.walk(root, followlinks=False):
        relative_dir = os.path.relpath(current, root)
        depth = 0 if relative_dir == "." else len(relative_dir.split(os.sep))
        if depth > max_depth:
            raise ValueError("generated directory tree is too deep")
        directories += len(dirs)
        if directories > max_directories:
            raise ValueError("too many generated directories")
        for name in dirs:
            path = os.path.join(current, name)
            if os.path.islink(path):
                raise ValueError(f"symlinked directory: {name}")
        for name in files:
            path = os.path.join(current, name)
            info = os.lstat(path)
            if not stat.S_ISREG(info.st_mode):
                raise ValueError(f"non-regular file: {name}")
            relative = os.path.relpath(path, root).replace(os.sep, "/")
            if len(relative) > max_path:
                raise ValueError("generated path is too long")
            entries[relative] = info.st_size
            total += info.st_size
            if len(entries) > max_entries:
                raise ValueError("too many generated files")
            if info.st_size > max(max_file, max_notebook):
                raise ValueError(f"generated file too large: {relative}")
            if total > max_total + max_notebook:
                raise ValueError("generated files are too large")
    encoded = json.dumps({"files": entries}, separators=(",", ":")).encode()
    if len(encoded) > 131072:
        raise ValueError("artifact inventory is too large")
    with open(output, "wb") as stream:
        stream.write(encoded)
except Exception as exc:
    print(str(exc), file=sys.stderr)
    raise SystemExit(2)
""".strip()


class MicrosandboxNotebookBackend(NotebookBackend):
    """Open explicitly selected notebook environments in local microVMs."""

    name = "microsandbox"

    def available(self) -> bool:
        return importlib.util.find_spec("microsandbox") is not None

    async def open(
        self,
        inputs: Sequence[NotebookInput],
        *,
        image: str,
        limits: ExecutionLimits,
    ) -> MicrosandboxNotebookEnvironment:
        validated = validate_inputs(inputs, limits, backend=self.name)
        cpu_count = _microsandbox_cpus(limits.cpu_cores)
        sdk = _load_microsandbox()
        name = f"sqlsaber-notebook-{uuid.uuid4().hex}"
        sandbox: Any | None = None
        try:
            async with asyncio.timeout(limits.image_prepare_seconds):
                sandbox = await sdk.Sandbox.create(
                    name,
                    image=image,
                    cpus=cpu_count,
                    memory=limits.memory_mb,
                    network=sdk.Network.none(),
                    security="restricted",
                    ephemeral=True,
                    max_duration=float(_PLATFORM_MAX_DURATION_SECONDS),
                )
        except TimeoutError as exc:
            await _best_effort_cleanup(sdk, name, sandbox)
            raise NotebookExecutionTimeout(
                "Microsandbox image preparation timed out after "
                f"{limits.image_prepare_seconds} seconds",
                backend=self.name,
                phase="image-prepare",
            ) from exc
        except asyncio.CancelledError:
            await _best_effort_cleanup(sdk, name, sandbox)
            raise
        except Exception as exc:
            await _best_effort_cleanup(sdk, name, sandbox)
            if _is_sdk_exception(
                exc, sdk, "ImageNotFoundError", "ImagePullFailedError"
            ) or _looks_like_image_error(exc):
                raise NotebookImageError(
                    f"Could not prepare Microsandbox notebook image {image!r}",
                    backend=self.name,
                    phase="image-prepare",
                    diagnostics=bound_log(str(exc), limits.max_log_chars),
                ) from exc
            raise NotebookBackendUnavailable(
                "Microsandbox could not start; verify this host supports Apple "
                "Virtualization, KVM, or Windows Hypervisor Platform",
                backend=self.name,
                phase="environment-open",
                diagnostics=bound_log(str(exc), limits.max_log_chars),
            ) from exc

        environment = MicrosandboxNotebookEnvironment(
            sdk=sdk,
            sandbox=sandbox,
            name=name,
            limits=limits,
        )
        try:
            async with asyncio.timeout(limits.open_seconds):
                await environment.stage(validated)
        except TimeoutError as exc:
            await _close_preserving_primary(environment)
            raise NotebookExecutionTimeout(
                "Microsandbox input staging timed out after "
                f"{limits.open_seconds} seconds",
                backend=self.name,
                phase="input-upload",
            ) from exc
        except BaseException:
            await _close_preserving_primary(environment)
            raise
        return environment


class MicrosandboxNotebookEnvironment(NotebookEnvironment):
    """A private Microsandbox VM reused for fresh-kernel notebook runs."""

    def __init__(
        self,
        *,
        sdk: Any,
        sandbox: Any,
        name: str,
        limits: ExecutionLimits,
    ) -> None:
        self.sdk = sdk
        self.sandbox = sandbox
        self.name = name
        self.limits = limits
        self.inputs_path = _INPUTS_PATH
        self.run_path = _RUN_PATH
        self._inventory: tuple[ArtifactInfo, ...] = ()
        self._lock = asyncio.Lock()
        self._closed = False
        self._cleanup_task: asyncio.Task[None] | None = None

    async def stage(self, inputs: Sequence[NotebookInput]) -> None:
        try:
            result = await self._exec(
                "input-upload",
                "mkdir",
                ["-p", _INPUTS_PATH, _RUN_PATH, _CONTROL_PATH],
                timeout=self.limits.open_seconds,
            )
            self._require_success(
                result,
                "Could not create Microsandbox notebook directories",
                phase="input-upload",
            )
            sentinel = f"{_INPUTS_PATH}/.sqlsaber-preflight"
            await self._write_file(sentinel, b"immutable", phase="input-upload")
            for item in inputs:
                await self._write_file(
                    f"{_INPUTS_PATH}/{item.name}",
                    item.data,
                    phase="input-upload",
                )

            result = await self._exec(
                "input-upload",
                "chown",
                ["-R", "root:root", _INPUTS_PATH],
                timeout=self.limits.open_seconds,
            )
            self._require_success(
                result,
                "Could not protect Microsandbox notebook input ownership",
                phase="input-upload",
            )
            result = await self._exec(
                "input-upload",
                "chmod",
                ["-R", "a-w", _INPUTS_PATH],
                timeout=self.limits.open_seconds,
            )
            self._require_success(
                result,
                "Could not make Microsandbox notebook inputs read-only",
                phase="input-upload",
            )
            result = await self._exec(
                "input-upload",
                "chown",
                ["-R", _RUN_OWNER, _RUN_PATH],
                timeout=self.limits.open_seconds,
            )
            self._require_success(
                result,
                "Could not prepare the Microsandbox run directory",
                phase="input-upload",
            )
            await self._preflight_runtime(sentinel)
            result = await self._exec(
                "input-upload",
                "rm",
                ["-f", sentinel],
                timeout=self.limits.open_seconds,
            )
            self._require_success(
                result,
                "Could not remove the Microsandbox input preflight sentinel",
                phase="input-upload",
            )
        except NotebookExecutionError:
            raise
        except Exception as exc:
            raise NotebookInfrastructureError(
                "Could not stage Microsandbox notebook inputs",
                backend="microsandbox",
                phase="input-upload",
                diagnostics=bound_log(str(exc), self.limits.max_log_chars),
            ) from exc

    async def execute(
        self,
        notebook: bytes,
        *,
        cell_timeout: int | None,
        command_timeout: int | None,
    ) -> NotebookExecutionResult:
        async with self._lock:
            self._ensure_open()
            validate_notebook_bytes(
                notebook,
                self.limits,
                backend="microsandbox",
                phase="notebook-upload",
            )
            self._inventory = ()
            try:
                await self._reset_run(notebook)
                timeout = _bounded_timeout(command_timeout, self.limits.command_seconds)
                returncode, stdout_bytes, stderr_bytes = await self._stream_exec(
                    "jupyter",
                    [
                        "nbconvert",
                        "--to",
                        "notebook",
                        "--execute",
                        "--inplace",
                        "--allow-errors",
                        "--ExecutePreprocessor.timeout="
                        f"{_bounded_timeout(cell_timeout, self.limits.cell_seconds) or -1}",
                        "notebook.ipynb",
                    ],
                    timeout=timeout,
                    cwd=_RUN_PATH,
                    user=_RUN_USER,
                    rlimits=[self.sdk.Rlimit.nproc(self.limits.pids)],
                )
                stdout = bound_log(stdout_bytes, self.limits.max_log_chars)
                stderr = bound_log(stderr_bytes, self.limits.max_log_chars)
                if returncode != 0:
                    raise NotebookInfrastructureError(
                        "Microsandbox notebook execution failed",
                        backend="microsandbox",
                        phase="notebook-execution",
                        diagnostics=stderr or stdout,
                    )
                await self._stop_run_processes()
                await self._freeze_run()
                sizes = await self._inventory_sizes()
                notebook_size = sizes.pop("notebook.ipynb", None)
                if notebook_size is None:
                    raise NotebookInfrastructureError(
                        "Microsandbox execution did not return a notebook",
                        backend="microsandbox",
                        phase="notebook-download",
                    )
                if notebook_size > self.limits.max_notebook_bytes:
                    raise NotebookLimitExceeded(
                        f"Notebook exceeds {self.limits.max_notebook_bytes} bytes",
                        backend="microsandbox",
                        phase="notebook-download",
                    )
                executed = await self._read_file(
                    _NOTEBOOK_PATH,
                    expected_size=notebook_size,
                    byte_limit=self.limits.max_notebook_bytes,
                    phase="notebook-download",
                )
                validate_notebook_bytes(
                    executed,
                    self.limits,
                    backend="microsandbox",
                    phase="notebook-download",
                )
                inventory = build_artifact_inventory(
                    sizes,
                    self.limits,
                    backend="microsandbox",
                )
                self._inventory = inventory
                return NotebookExecutionResult(
                    notebook=executed,
                    artifacts=inventory,
                    stdout=stdout,
                    stderr=stderr,
                )
            except NotebookExecutionTimeout:
                await self._terminate_after_interruption()
                raise
            except asyncio.CancelledError:
                await self._terminate_after_interruption()
                raise
            except NotebookExecutionError:
                raise
            except Exception as exc:
                if _is_sdk_exception(exc, self.sdk, "ExecTimeoutError"):
                    await self._terminate_after_interruption()
                    raise NotebookExecutionTimeout(
                        "Microsandbox notebook execution timed out",
                        backend="microsandbox",
                        phase="notebook-execution",
                    ) from exc
                raise NotebookInfrastructureError(
                    "Could not transfer Microsandbox notebook results",
                    backend="microsandbox",
                    phase="notebook-download",
                    diagnostics=bound_log(str(exc), self.limits.max_log_chars),
                ) from exc

    async def read_artifact(self, artifact: ArtifactInfo) -> bytes:
        async with self._lock:
            self._ensure_open()
            if artifact not in self._inventory:
                raise NotebookInfrastructureError(
                    f"Unknown artifact: {artifact.path}",
                    backend="microsandbox",
                    phase="artifact-download",
                )
            relative = validate_artifact_path(artifact.path, backend="microsandbox")
            path = f"{_RUN_PATH}/{relative.as_posix()}"
            try:
                metadata = await self._operation(
                    self.sandbox.fs.stat(path),
                    timeout=self.limits.open_seconds,
                    phase="artifact-download",
                )
                if metadata.kind != "file" or metadata.size != artifact.size:
                    raise ValueError("artifact changed after inventory")
                return await self._read_file(
                    path,
                    expected_size=artifact.size,
                    byte_limit=self.limits.max_artifact_bytes,
                    phase="artifact-download",
                )
            except NotebookExecutionTimeout:
                await self._terminate_after_interruption()
                raise
            except NotebookExecutionError:
                raise
            except Exception as exc:
                raise NotebookInfrastructureError(
                    f"Could not read artifact: {artifact.path}",
                    backend="microsandbox",
                    phase="artifact-download",
                    diagnostics=bound_log(str(exc), self.limits.max_log_chars),
                ) from exc

    async def list_workspace(self) -> tuple[ArtifactInfo, ...]:
        async with self._lock:
            self._ensure_open()
            return self._inventory

    async def close(self) -> None:
        async with self._lock:
            if self._cleanup_task is None:
                self._closed = True
                self._inventory = ()
                self._cleanup_task = asyncio.create_task(self._cleanup())
            task = self._cleanup_task
        await asyncio.shield(task)

    async def _reset_run(self, notebook: bytes) -> None:
        result = await self._exec(
            "run-setup",
            "rm",
            ["-rf", _RUN_PATH],
            timeout=self.limits.open_seconds,
        )
        self._require_success(
            result,
            "Could not reset the Microsandbox run directory",
            phase="run-setup",
        )
        result = await self._exec(
            "run-setup",
            "mkdir",
            ["-p", _RUN_PATH],
            timeout=self.limits.open_seconds,
        )
        self._require_success(
            result,
            "Could not recreate the Microsandbox run directory",
            phase="run-setup",
        )
        await self._write_file(_NOTEBOOK_PATH, notebook, phase="notebook-upload")
        result = await self._exec(
            "run-setup",
            "chown",
            ["-R", _RUN_OWNER, _RUN_PATH],
            timeout=self.limits.open_seconds,
        )
        self._require_success(
            result,
            "Notebook image cannot prepare an unprivileged run directory",
            phase="run-setup",
            image_error=True,
        )

    async def _stop_run_processes(self) -> None:
        result = await self._exec(
            "notebook-execution",
            _PYTHON,
            ["-c", _STOP_USER_PROCESSES_SCRIPT],
            timeout=self.limits.open_seconds,
        )
        self._require_success(
            result,
            "Could not stop background notebook processes",
            phase="notebook-execution",
        )

    async def _freeze_run(self) -> None:
        result = await self._exec(
            "artifact-inventory",
            "chown",
            ["-R", "root:root", _RUN_PATH],
            timeout=self.limits.open_seconds,
        )
        self._require_success(
            result,
            "Could not freeze Microsandbox notebook results",
            phase="artifact-inventory",
        )
        result = await self._exec(
            "artifact-inventory",
            "chmod",
            ["-R", "a-w", _RUN_PATH],
            timeout=self.limits.open_seconds,
        )
        self._require_success(
            result,
            "Could not freeze Microsandbox notebook results",
            phase="artifact-inventory",
        )

    async def _preflight_runtime(self, sentinel: str) -> None:
        result = await self._exec(
            "image-preflight",
            "jupyter",
            ["nbconvert", "--version"],
            timeout=self.limits.open_seconds,
            user=_RUN_USER,
        )
        self._require_success(
            result,
            "Notebook image must provide UID 1000, jupyter, and nbconvert",
            phase="image-preflight",
            image_error=True,
        )
        result = await self._exec(
            "image-preflight",
            _PYTHON,
            ["-c", _PREFLIGHT_SCRIPT, sentinel, _RUN_PATH],
            timeout=self.limits.open_seconds,
            user=_RUN_USER,
        )
        self._require_success(
            result,
            "Notebook image does not preserve immutable inputs for UID 1000",
            phase="image-preflight",
            image_error=True,
        )

    async def _inventory_sizes(self) -> dict[str, int]:
        with contextlib.suppress(Exception):
            await self._operation(
                self.sandbox.fs.remove(_INVENTORY_PATH),
                timeout=self.limits.open_seconds,
                phase="artifact-inventory",
            )
        result = await self._exec(
            "artifact-inventory",
            _PYTHON,
            [
                "-c",
                _INVENTORY_SCRIPT,
                _RUN_PATH,
                _INVENTORY_PATH,
                str(self.limits.max_artifacts),
                str(self.limits.max_artifact_bytes),
                str(self.limits.max_total_artifact_bytes),
                str(self.limits.max_notebook_bytes),
            ],
            timeout=self.limits.open_seconds,
        )
        if not result.success:
            raise NotebookLimitExceeded(
                "Microsandbox artifact inventory exceeded limits or contained unsafe files",
                backend="microsandbox",
                phase="artifact-inventory",
                diagnostics=bound_log(
                    result.stderr_bytes or result.stdout_bytes,
                    self.limits.max_log_chars,
                ),
            )
        metadata = await self._operation(
            self.sandbox.fs.stat(_INVENTORY_PATH),
            timeout=self.limits.open_seconds,
            phase="artifact-inventory",
        )
        if metadata.kind != "file" or metadata.size > _INVENTORY_MAX_BYTES:
            raise NotebookInfrastructureError(
                "Microsandbox returned an invalid artifact inventory",
                backend="microsandbox",
                phase="artifact-inventory",
            )
        payload_bytes = await self._read_file(
            _INVENTORY_PATH,
            expected_size=metadata.size,
            byte_limit=_INVENTORY_MAX_BYTES,
            phase="artifact-inventory",
        )
        try:
            payload = json.loads(payload_bytes)
            files = payload["files"]
            if not isinstance(files, dict):
                raise TypeError("files must be an object")
            parsed: dict[str, int] = {}
            for path, size in files.items():
                if not isinstance(path, str) or not isinstance(size, int):
                    raise TypeError("inventory entries must be string/integer pairs")
                parsed[path] = size
            return parsed
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise NotebookInfrastructureError(
                "Microsandbox returned an invalid artifact inventory",
                backend="microsandbox",
                phase="artifact-inventory",
            ) from exc

    async def _write_file(self, path: str, data: bytes, *, phase: str) -> None:
        try:
            async with asyncio.timeout(self.limits.open_seconds):
                sink = await self.sandbox.fs.write_stream(path)
                async with sink:
                    for offset in range(0, len(data), _STREAM_CHUNK_BYTES):
                        await sink.write(data[offset : offset + _STREAM_CHUNK_BYTES])
        except TimeoutError as exc:
            raise NotebookExecutionTimeout(
                f"Microsandbox {phase} timed out after "
                f"{self.limits.open_seconds} seconds",
                backend="microsandbox",
                phase=phase,
            ) from exc

    async def _read_file(
        self,
        path: str,
        *,
        expected_size: int,
        byte_limit: int,
        phase: str,
    ) -> bytes:
        if expected_size < 0 or expected_size > byte_limit:
            raise NotebookLimitExceeded(
                f"Microsandbox file exceeds {byte_limit} bytes",
                backend="microsandbox",
                phase=phase,
            )
        stream = await self._operation(
            self.sandbox.fs.read_stream(path),
            timeout=self.limits.open_seconds,
            phase=phase,
        )
        data = bytearray()
        try:
            async with asyncio.timeout(self.limits.open_seconds):
                async for chunk in stream:
                    data.extend(chunk)
                    if len(data) > expected_size or len(data) > byte_limit:
                        raise NotebookInfrastructureError(
                            "Microsandbox file changed during download",
                            backend="microsandbox",
                            phase=phase,
                        )
        except TimeoutError as exc:
            raise NotebookExecutionTimeout(
                f"Microsandbox {phase} timed out after "
                f"{self.limits.open_seconds} seconds",
                backend="microsandbox",
                phase=phase,
            ) from exc
        if len(data) != expected_size:
            raise NotebookInfrastructureError(
                "Microsandbox file changed during download",
                backend="microsandbox",
                phase=phase,
            )
        return bytes(data)

    async def _exec(
        self,
        phase: str,
        command: str,
        args: list[str],
        *,
        timeout: int | None,
        user: str = "root",
    ) -> Any:
        try:
            async with asyncio.timeout(timeout):
                return await self.sandbox.exec(
                    command,
                    args,
                    user=user,
                    timeout=timeout,
                )
        except TimeoutError as exc:
            raise NotebookExecutionTimeout(
                f"Microsandbox {phase} timed out after {timeout} seconds",
                backend="microsandbox",
                phase=phase,
            ) from exc
        except Exception as exc:
            if _is_sdk_exception(exc, self.sdk, "ExecTimeoutError"):
                raise NotebookExecutionTimeout(
                    f"Microsandbox {phase} timed out after {timeout} seconds",
                    backend="microsandbox",
                    phase=phase,
                ) from exc
            raise

    async def _stream_exec(
        self,
        command: str,
        args: list[str],
        *,
        timeout: int | None,
        cwd: str,
        user: str,
        rlimits: list[Any],
    ) -> tuple[int, bytes, bytes]:
        handle: Any | None = None
        stdout = _BoundedBytes(self.limits.max_log_chars)
        stderr = _BoundedBytes(self.limits.max_log_chars)
        exit_code: int | None = None
        try:
            async with asyncio.timeout(timeout):
                handle = await self.sandbox.exec_stream(
                    command,
                    args,
                    cwd=cwd,
                    user=user,
                    env={"HOME": "/tmp"},
                    timeout=timeout,
                    rlimits=rlimits,
                )
                async for event in handle:
                    if event.event_type == "stdout" and event.data:
                        stdout.append(event.data)
                    elif event.event_type in {"stderr", "failed"} and event.data:
                        stderr.append(event.data)
                    elif event.event_type == "exited":
                        exit_code = event.code
                if exit_code is None:
                    exit_code, _ = await handle.wait()
        except TimeoutError as exc:
            await _kill_exec_handle(handle)
            raise NotebookExecutionTimeout(
                f"Microsandbox notebook execution timed out after {timeout} seconds",
                backend="microsandbox",
                phase="notebook-execution",
            ) from exc
        except asyncio.CancelledError:
            await _kill_exec_handle(handle)
            raise
        except Exception as exc:
            await _kill_exec_handle(handle)
            if _is_sdk_exception(exc, self.sdk, "ExecTimeoutError"):
                raise NotebookExecutionTimeout(
                    f"Microsandbox notebook execution timed out after {timeout} seconds",
                    backend="microsandbox",
                    phase="notebook-execution",
                ) from exc
            raise
        return exit_code or 0, stdout.value(), stderr.value()

    async def _operation(self, operation: Any, *, timeout: int, phase: str) -> Any:
        try:
            async with asyncio.timeout(timeout):
                return await operation
        except TimeoutError as exc:
            raise NotebookExecutionTimeout(
                f"Microsandbox {phase} timed out after {timeout} seconds",
                backend="microsandbox",
                phase=phase,
            ) from exc

    def _require_success(
        self,
        result: Any,
        message: str,
        *,
        phase: str,
        image_error: bool = False,
    ) -> None:
        if result.success:
            return
        error_type = NotebookImageError if image_error else NotebookInfrastructureError
        raise error_type(
            message,
            backend="microsandbox",
            phase=phase,
            diagnostics=bound_log(
                result.stderr_bytes or result.stdout_bytes,
                self.limits.max_log_chars,
            ),
        )

    async def _terminate_after_interruption(self) -> None:
        self._closed = True
        self._inventory = ()
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup())
        with contextlib.suppress(Exception):
            await asyncio.shield(self._cleanup_task)

    async def _cleanup(self) -> None:
        try:
            await _cleanup_named_sandbox(self.sdk, self.name, self.sandbox)
        except Exception as exc:
            raise NotebookInfrastructureError(
                "Could not remove the Microsandbox notebook environment",
                backend="microsandbox",
                phase="cleanup",
                diagnostics=bound_log(str(exc), self.limits.max_log_chars),
            ) from exc

    def _ensure_open(self) -> None:
        if self._closed:
            raise NotebookInfrastructureError(
                "Notebook environment is closed",
                backend="microsandbox",
                phase="lifecycle",
            )


class _BoundedBytes:
    def __init__(self, log_chars: int) -> None:
        self._limit = max(log_chars * 4, 32_000)
        self._head_limit = self._limit // 2
        self._tail_limit = self._limit - self._head_limit
        self._head = bytearray()
        self._tail = bytearray()

    def append(self, chunk: bytes) -> None:
        if len(self._head) < self._head_limit:
            consumed = min(self._head_limit - len(self._head), len(chunk))
            self._head.extend(chunk[:consumed])
            chunk = chunk[consumed:]
        if chunk:
            self._tail.extend(chunk)
            if len(self._tail) > self._tail_limit:
                del self._tail[: -self._tail_limit]

    def value(self) -> bytes:
        return bytes(self._head + self._tail)


def _microsandbox_cpus(value: float) -> int:
    if not math.isfinite(value):
        raise NotebookLimitExceeded(
            "Microsandbox CPU limit must be finite",
            backend="microsandbox",
            phase="configuration",
        )
    result = math.floor(value)
    if result < 1:
        raise NotebookLimitExceeded(
            "Microsandbox requires at least one whole CPU",
            backend="microsandbox",
            phase="configuration",
        )
    return result


def _bounded_timeout(
    requested: int | None,
    configured: int | None,
) -> int | None:
    if requested is None:
        return configured
    if configured is None:
        return requested
    return min(requested, configured)


async def _kill_exec_handle(handle: Any | None) -> None:
    if handle is None:
        return
    with contextlib.suppress(Exception):
        await asyncio.wait_for(handle.kill(), timeout=5)


async def _cleanup_named_sandbox(sdk: Any, name: str, sandbox: Any | None) -> None:
    if sandbox is not None:
        with contextlib.suppress(Exception):
            await asyncio.wait_for(sandbox.kill(timeout=10.0), timeout=15)

    try:
        handle = await asyncio.wait_for(sdk.Sandbox.get(name), timeout=10)
    except Exception as exc:
        if _is_sdk_exception(exc, sdk, "SandboxNotFoundError"):
            return
        raise

    if str(handle.status).lower() in {"running", "draining", "paused"}:
        with contextlib.suppress(Exception):
            await asyncio.wait_for(handle.kill(timeout=10.0), timeout=15)
        handle = await asyncio.wait_for(handle.refresh(), timeout=10)
    await asyncio.wait_for(handle.remove(), timeout=10)
    try:
        await asyncio.wait_for(sdk.Sandbox.get(name), timeout=10)
    except Exception as exc:
        if _is_sdk_exception(exc, sdk, "SandboxNotFoundError"):
            return
        raise
    raise RuntimeError(f"Microsandbox sandbox {name!r} still exists after removal")


async def _best_effort_cleanup(sdk: Any, name: str, sandbox: Any | None) -> None:
    task = asyncio.create_task(_cleanup_named_sandbox(sdk, name, sandbox))
    with contextlib.suppress(Exception):
        await asyncio.shield(task)


async def _close_preserving_primary(
    environment: MicrosandboxNotebookEnvironment,
) -> None:
    with contextlib.suppress(Exception):
        await environment.close()


def _looks_like_image_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "image error:" in message or "registry error:" in message


def _is_sdk_exception(exc: Exception, sdk: Any, *names: str) -> bool:
    types = tuple(
        candidate
        for name in names
        if isinstance((candidate := getattr(sdk, name, None)), type)
    )
    return bool(types) and isinstance(exc, types)


def _load_microsandbox() -> Any:
    try:
        return importlib.import_module("microsandbox")
    except ImportError as exc:
        raise NotebookBackendUnavailable(
            "Microsandbox backend requires the `sqlsaber-notebook[microsandbox]` extra",
            backend="microsandbox",
            phase="availability",
        ) from exc
