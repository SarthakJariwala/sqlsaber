"""E2B notebook sandboxes built from the configured Jupyter image."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import importlib.util
import json
import math
import shlex
from collections.abc import AsyncIterator, Sequence
from typing import Any

from ._files import build_artifact_inventory, validate_notebook_bytes
from .base import (
    ArtifactInfo,
    ExecutionLimits,
    NotebookBackendUnavailable,
    NotebookExecutionError,
    NotebookExecutionResult,
    NotebookExecutionTimeout,
    NotebookImageError,
    NotebookInfrastructureError,
    NotebookInput,
    NotebookLimitExceeded,
    bound_log,
    validate_inputs,
)

_ROOT = "/root/sqlsaber-notebook"
_LIFETIME_SECONDS = 3600  # Available on Hobby as well as paid E2B plans.

# The controller runs as root, outside the notebook-writable directory. Logs
# stay remote and only bounded diagnostics and a bounded inventory cross the API.
_RUN_SCRIPT = r"""
import json
import os
import pathlib
import pwd
import signal
import stat
import subprocess
import sys
import time

root = pathlib.Path(sys.argv[1])
limits = json.loads(sys.argv[2])
run = root / "run"
uid = pwd.getpwnam("jovyan").pw_uid
with (root / "stdout").open("wb") as out, (root / "stderr").open("wb") as err:
    result = subprocess.run([
        "/usr/sbin/runuser", "-u", "jovyan", "--",
        "/opt/conda/bin/jupyter", "nbconvert", "--to", "notebook",
        "--execute", "--inplace", "--allow-errors",
        "--ExecutePreprocessor.timeout=" + str(limits["cell"]),
        "notebook.ipynb",
    ], cwd=run, stdout=out, stderr=err)

# Stop background children before inspecting or downloading notebook output.
for attempt in range(20):
    found = False
    for entry in pathlib.Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid == uid:
                os.kill(int(entry.name), signal.SIGKILL)
                found = True
        except (FileNotFoundError, ProcessLookupError):
            pass
    if not found:
        break
    time.sleep(0.05)

def log(name):
    with (root / name).open("rb") as stream:
        size = os.fstat(stream.fileno()).st_size
        half = limits["log"] // 2
        head = stream.read(half)
        stream.seek(max(half, size - half))
        return (head + stream.read(half)).decode("utf-8", errors="replace")

payload = {"stdout": log("stdout"), "stderr": log("stderr"), "code": result.returncode}
try:
    files = {}
    total = 0
    directories = 0
    for current, dirs, names in os.walk(run, followlinks=False):
        directories += len(dirs)
        if directories > 256 or len(pathlib.Path(current).relative_to(run).parts) > 32:
            raise ValueError("generated directory tree exceeds limits")
        for name in dirs:
            if os.path.islink(os.path.join(current, name)):
                raise ValueError("symlinked generated directory")
        for name in names:
            path = pathlib.Path(current) / name
            info = path.lstat()
            relative = path.relative_to(run).as_posix()
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError("non-regular or linked generated file")
            if len(relative) > 1024:
                raise ValueError("generated path is too long")
            maximum = limits["notebook"] if relative == "notebook.ipynb" else limits["file"]
            if info.st_size > maximum:
                raise ValueError("generated file exceeds byte limit")
            if relative != "notebook.ipynb":
                total += info.st_size
            files[relative] = info.st_size
            if len(files) > limits["count"] + 1 or total > limits["total"]:
                raise ValueError("generated artifacts exceed limits")
            os.chown(path, 0, 0)
            os.chmod(path, 0o444)
        os.chown(current, 0, 0)
        os.chmod(current, 0o555)
    payload["files"] = files
except (OSError, ValueError) as exc:
    payload["error"] = str(exc)
print(json.dumps(payload))
""".strip()


class E2BNotebookBackend:
    name = "e2b"

    def __init__(self, *, api_key: str | None = None) -> None:
        self._api_key = api_key

    def available(self) -> bool:
        return importlib.util.find_spec("e2b") is not None

    async def open(
        self,
        inputs: Sequence[NotebookInput],
        *,
        image: str,
        limits: ExecutionLimits,
    ) -> E2BNotebookEnvironment:
        validated = validate_inputs(inputs, limits, backend=self.name)
        try:
            from e2b import AsyncSandbox, AsyncTemplate, TimeoutException
        except ImportError as exc:
            raise NotebookBackendUnavailable(
                "E2B backend requires the `sqlsaber-notebook[e2b]` extra",
                backend=self.name,
                phase="availability",
            ) from exc
        options = {"api_key": self._api_key} if self._api_key is not None else {}
        cpu = math.ceil(limits.cpu_cores)
        identity = hashlib.sha256(
            f"v1:{image}:{cpu}:{limits.memory_mb}".encode()
        ).hexdigest()[:24]
        try:
            async with asyncio.timeout(limits.image_prepare_seconds):
                build = await AsyncTemplate.build(
                    AsyncTemplate().from_image(image).set_user("root"),
                    name=f"sqlsaber-notebook-{identity}",
                    cpu_count=cpu,
                    memory_mb=limits.memory_mb,
                    **options,
                )
        except (TimeoutError, TimeoutException) as exc:
            raise NotebookExecutionTimeout(
                "E2B template preparation timed out",
                backend=self.name,
                phase="image-prepare",
            ) from exc
        except Exception as exc:
            raise NotebookImageError(
                "Could not build E2B notebook template; check E2B_API_KEY and image access",
                backend=self.name,
                phase="image-prepare",
                diagnostics=bound_log(str(exc), limits.max_log_chars),
            ) from exc
        try:
            async with asyncio.timeout(limits.open_seconds):
                sandbox = await AsyncSandbox.create(
                    template=build.template_id,
                    timeout=_LIFETIME_SECONDS,
                    allow_internet_access=False,
                    **options,
                )
        except (TimeoutError, TimeoutException) as exc:
            raise NotebookExecutionTimeout(
                "E2B sandbox creation timed out",
                backend=self.name,
                phase="environment-open",
            ) from exc
        except Exception as exc:
            raise NotebookBackendUnavailable(
                "Could not create E2B notebook sandbox; check E2B_API_KEY and account limits",
                backend=self.name,
                phase="environment-open",
                diagnostics=bound_log(str(exc), limits.max_log_chars),
            ) from exc
        environment = E2BNotebookEnvironment(sandbox, limits)
        try:
            async with _operation(limits.open_seconds, "input-upload"):
                await environment._command(
                    f"chmod 711 /root && mkdir -p {_ROOT}/inputs {_ROOT}/run",
                    timeout=limits.open_seconds,
                    phase="input-upload",
                )
                for item in validated:
                    await environment._write(f"{_ROOT}/inputs/{item.name}", item.data)
                await environment._command(
                    f"chmod 755 {_ROOT} && chmod -R a-w {_ROOT}/inputs && "
                    "/usr/sbin/runuser -u jovyan -- /opt/conda/bin/jupyter nbconvert --version",
                    timeout=limits.open_seconds,
                    phase="image-preflight",
                )
                await environment._write(f"{_ROOT}/runner.py", _RUN_SCRIPT.encode())
        except BaseException:
            with contextlib.suppress(NotebookExecutionError):
                await environment.close()
            raise
        return environment


class E2BNotebookEnvironment:
    def __init__(self, sandbox: Any, limits: ExecutionLimits) -> None:
        self.sandbox = sandbox
        self.limits = limits
        self._inventory: tuple[ArtifactInfo, ...] = ()
        self._lock = asyncio.Lock()
        self._closed = False
        self._cleanup_task: asyncio.Task[None] | None = None

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
                notebook, self.limits, backend="e2b", phase="notebook-upload"
            )
            self._inventory = ()
            cells = [
                t for t in (cell_timeout, self.limits.cell_seconds) if t is not None
            ]
            commands = [
                t
                for t in (command_timeout, self.limits.command_seconds)
                if t is not None
            ]
            budget = {
                "cell": min(cells) if cells else -1,
                "notebook": self.limits.max_notebook_bytes,
                "file": self.limits.max_artifact_bytes,
                "count": self.limits.max_artifacts,
                "total": self.limits.max_total_artifact_bytes,
                "log": self.limits.max_log_chars,
            }
            try:
                await self._command(
                    f"rm -rf {_ROOT}/run && mkdir {_ROOT}/run",
                    timeout=self.limits.open_seconds,
                )
                await self._write(f"{_ROOT}/run/notebook.ipynb", notebook)
                await self._command(
                    f"chown -R jovyan:users {_ROOT}/run",
                    timeout=self.limits.open_seconds,
                )
                output = await self._command(
                    shlex.join(
                        [
                            "/usr/bin/python3",
                            f"{_ROOT}/runner.py",
                            _ROOT,
                            json.dumps(budget),
                        ]
                    ),
                    timeout=min(commands) if commands else None,
                )
                payload = json.loads(output)
                if payload["code"]:
                    raise NotebookInfrastructureError(
                        "E2B notebook execution failed",
                        backend="e2b",
                        phase="notebook-execution",
                        diagnostics=bound_log(
                            payload["stderr"], self.limits.max_log_chars
                        ),
                    )
                if "error" in payload:
                    raise NotebookLimitExceeded(
                        payload["error"], backend="e2b", phase="artifact-inventory"
                    )
                sizes = payload["files"]
                size = sizes.pop("notebook.ipynb")
                inventory = build_artifact_inventory(sizes, self.limits, backend="e2b")
                executed = await self._read("notebook.ipynb", size)
                validate_notebook_bytes(
                    executed, self.limits, backend="e2b", phase="notebook-download"
                )
                self._inventory = inventory
                return NotebookExecutionResult(
                    executed,
                    inventory,
                    bound_log(payload["stdout"], self.limits.max_log_chars),
                    bound_log(payload["stderr"], self.limits.max_log_chars),
                )
            except BaseException as exc:
                with contextlib.suppress(NotebookExecutionError):
                    await self._terminate()
                if isinstance(exc, (NotebookExecutionError, asyncio.CancelledError)):
                    raise
                raise NotebookInfrastructureError(
                    "Could not execute or transfer E2B notebook results",
                    backend="e2b",
                    phase="notebook-execution",
                    diagnostics=bound_log(str(exc), self.limits.max_log_chars),
                ) from exc

    async def read_artifact(self, artifact: ArtifactInfo) -> bytes:
        async with self._lock:
            self._ensure_open()
            if artifact not in self._inventory:
                raise NotebookInfrastructureError(
                    f"Unknown artifact: {artifact.path}",
                    backend="e2b",
                    phase="artifact-download",
                )
            return await self._read(artifact.path, artifact.size)

    async def list_workspace(self) -> tuple[ArtifactInfo, ...]:
        async with self._lock:
            self._ensure_open()
            return self._inventory

    async def close(self) -> None:
        async with self._lock:
            await self._terminate()

    async def _terminate(self) -> None:
        self._closed = True
        self._inventory = ()
        if self._cleanup_task is None or (
            self._cleanup_task.done() and self._cleanup_task.exception() is not None
        ):
            self._cleanup_task = asyncio.create_task(self._kill())
        await asyncio.shield(self._cleanup_task)

    async def _kill(self) -> None:
        async with _operation(30, "cleanup"):
            await self.sandbox.kill(request_timeout=30)

    def _ensure_open(self) -> None:
        if self._closed:
            raise NotebookInfrastructureError(
                "Notebook environment is closed", backend="e2b", phase="lifecycle"
            )

    async def _command(
        self, command: str, *, timeout: int | None, phase: str = "notebook-execution"
    ) -> str:
        async with _operation(timeout, phase):
            result = await self.sandbox.commands.run(
                command,
                user="root",
                timeout=timeout or 0,
                request_timeout=self.limits.open_seconds,
            )
            return result.stdout

    async def _write(self, path: str, data: bytes) -> None:
        async with _operation(self.limits.open_seconds, "input-upload"):
            await self.sandbox.files.write(
                path, data, user="root", request_timeout=self.limits.open_seconds
            )

    async def _read(self, name: str, expected: int) -> bytes:
        async with _operation(self.limits.open_seconds, "artifact-download"):
            data = bytes(
                await self.sandbox.files.read(
                    f"{_ROOT}/run/{name}",
                    format="bytes",
                    user="root",
                    request_timeout=self.limits.open_seconds,
                )
            )
        if len(data) != expected:
            raise NotebookInfrastructureError(
                "E2B file changed after inventory",
                backend="e2b",
                phase="artifact-download",
            )
        return data


@contextlib.asynccontextmanager
async def _operation(timeout: int | None, phase: str) -> AsyncIterator[None]:
    from e2b import TimeoutException

    try:
        async with asyncio.timeout(timeout):
            yield
    except (TimeoutError, TimeoutException) as exc:
        raise NotebookExecutionTimeout(
            f"E2B {phase} timed out", backend="e2b", phase=phase
        ) from exc
    except NotebookExecutionError:
        raise
    except Exception as exc:
        error = (
            NotebookImageError
            if phase == "image-preflight"
            else NotebookInfrastructureError
        )
        raise error(f"E2B {phase} failed", backend="e2b", phase=phase) from exc
