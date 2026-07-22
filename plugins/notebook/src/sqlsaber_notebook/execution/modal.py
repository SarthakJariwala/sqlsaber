"""Remote Modal Sandbox implementation of notebook execution."""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import json
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

_APP_NAME = "sqlsaber-notebook"
_REMOTE_BASE = "/tmp/sqlsaber-notebook"
_RUN_USER = "jovyan"
_RUN_GROUP = "users"
_RUNUSER = "/usr/sbin/runuser"
_PYTHON = "/opt/conda/bin/python"
# Modal requires a finite Sandbox lifetime; use its 24-hour platform ceiling
# rather than imposing a SQLsaber analysis timeout.
_MODAL_PLATFORM_TIMEOUT_SECONDS = 24 * 60 * 60

_INVENTORY_SCRIPT = r"""
import json
import os
import stat
import sys

root = sys.argv[1]
max_entries = int(sys.argv[2]) + 1  # include notebook.ipynb
max_file = int(sys.argv[3])
max_total = int(sys.argv[4])
max_notebook = int(sys.argv[5])
entries = {}
total = 0
try:
    for current, dirs, files in os.walk(root, followlinks=False):
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
            entries[relative] = info.st_size
            total += info.st_size
            if len(entries) > max_entries:
                raise ValueError("too many generated files")
            if info.st_size > max(max_file, max_notebook):
                raise ValueError(f"generated file too large: {relative}")
            if total > max_total + max_notebook:
                raise ValueError("generated files too large")
except Exception as exc:
    print(json.dumps({"error": str(exc)}))
    raise SystemExit(2)
print(json.dumps({"files": entries}, separators=(",", ":")))
""".strip()


class ModalNotebookBackend(NotebookBackend):
    """Open explicitly selected notebook environments in Modal."""

    name = "modal"

    def available(self) -> bool:
        return importlib.util.find_spec("modal") is not None

    async def open(
        self,
        inputs: Sequence[NotebookInput],
        *,
        image: str,
        limits: ExecutionLimits,
    ) -> ModalNotebookEnvironment:
        validated = validate_inputs(inputs, limits, backend=self.name)
        modal = _load_modal()
        try:
            async with asyncio.timeout(limits.open_seconds):
                app = await modal.App.lookup.aio(
                    _APP_NAME,
                    create_if_missing=True,
                )
                runtime_image = modal.Image.from_registry(image)
                sandbox = await modal.Sandbox.create.aio(
                    "sleep",
                    "infinity",
                    app=app,
                    image=runtime_image,
                    block_network=True,
                    cpu=limits.cpu_cores,
                    memory=limits.memory_mb,
                    timeout=_MODAL_PLATFORM_TIMEOUT_SECONDS,
                )
        except TimeoutError as exc:
            raise NotebookExecutionTimeout(
                f"Modal environment open timed out after {limits.open_seconds} seconds",
                backend=self.name,
                phase="environment-open",
            ) from exc
        except Exception as exc:
            raise NotebookBackendUnavailable(
                "Modal is unavailable; run `modal setup` and verify the active profile",
                backend=self.name,
                phase="environment-open",
                diagnostics=bound_log(str(exc), limits.max_log_chars),
            ) from exc

        environment = ModalNotebookEnvironment(
            sandbox=sandbox,
            limits=limits,
            remote_root=f"{_REMOTE_BASE}-{uuid.uuid4().hex}",
        )
        try:
            async with asyncio.timeout(limits.open_seconds):
                await environment.stage(validated)
        except TimeoutError as exc:
            await environment.close()
            raise NotebookExecutionTimeout(
                f"Modal input staging timed out after {limits.open_seconds} seconds",
                backend=self.name,
                phase="input-upload",
            ) from exc
        except BaseException:
            await environment.close()
            raise
        return environment


class ModalNotebookEnvironment(NotebookEnvironment):
    def __init__(
        self, *, sandbox: Any, limits: ExecutionLimits, remote_root: str
    ) -> None:
        self.sandbox = sandbox
        self.limits = limits
        self.remote_root = remote_root
        self.inputs_path = f"{remote_root}/inputs"
        self.run_path = f"{remote_root}/run"
        self._inventory: tuple[ArtifactInfo, ...] = ()
        self._lock = asyncio.Lock()
        self._closed = False

    async def stage(self, inputs: Sequence[NotebookInput]) -> None:
        try:
            await _modal_operation(
                self.sandbox.filesystem.make_directory.aio(
                    self.inputs_path,
                    create_parents=True,
                ),
                timeout=self.limits.open_seconds,
                phase="input-upload",
            )
            await _modal_operation(
                self.sandbox.filesystem.make_directory.aio(
                    self.run_path,
                    create_parents=True,
                ),
                timeout=self.limits.open_seconds,
                phase="input-upload",
            )
            for item in inputs:
                await _modal_operation(
                    self.sandbox.filesystem.write_bytes.aio(
                        item.data,
                        f"{self.inputs_path}/{item.name}",
                    ),
                    timeout=self.limits.open_seconds,
                    phase="input-upload",
                )
            result = await self._exec(
                "stage-inputs",
                "chmod",
                "-R",
                "a-w",
                self.inputs_path,
                timeout=self.limits.open_seconds,
            )
            if result[0] != 0:
                raise NotebookInfrastructureError(
                    "Could not make Modal notebook inputs read-only",
                    backend="modal",
                    phase="input-upload",
                    diagnostics=result[2],
                )
            await self._preflight_runtime()
        except NotebookExecutionError:
            raise
        except Exception as exc:
            raise NotebookInfrastructureError(
                "Could not stage Modal notebook inputs",
                backend="modal",
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
                backend="modal",
                phase="notebook-upload",
            )
            try:
                await _modal_operation(
                    self.sandbox.filesystem.remove.aio(self.run_path, recursive=True),
                    timeout=self.limits.open_seconds,
                    phase="run-setup",
                )
                await _modal_operation(
                    self.sandbox.filesystem.make_directory.aio(
                        self.run_path,
                        create_parents=True,
                    ),
                    timeout=self.limits.open_seconds,
                    phase="run-setup",
                )
                await _modal_operation(
                    self.sandbox.filesystem.write_bytes.aio(
                        notebook,
                        f"{self.run_path}/notebook.ipynb",
                    ),
                    timeout=self.limits.open_seconds,
                    phase="notebook-upload",
                )
                ownership = await self._exec(
                    "run-setup",
                    "chown",
                    "-R",
                    f"{_RUN_USER}:{_RUN_GROUP}",
                    self.run_path,
                    timeout=self.limits.open_seconds,
                )
                if ownership[0] != 0:
                    raise NotebookImageError(
                        "Notebook image cannot prepare a non-root Modal run directory",
                        backend="modal",
                        phase="run-setup",
                        diagnostics=ownership[2],
                    )
                returncode, stdout, stderr = await self._exec(
                    "notebook-execution",
                    _RUNUSER,
                    "-u",
                    _RUN_USER,
                    "--",
                    "jupyter",
                    "nbconvert",
                    "--to",
                    "notebook",
                    "--execute",
                    "--inplace",
                    "--allow-errors",
                    "--ExecutePreprocessor.timeout="
                    f"{_bounded_timeout(cell_timeout, self.limits.cell_seconds) or -1}",
                    "notebook.ipynb",
                    timeout=_bounded_timeout(
                        command_timeout, self.limits.command_seconds
                    ),
                    workdir=self.run_path,
                )
                if returncode != 0:
                    raise NotebookInfrastructureError(
                        "Modal notebook execution failed",
                        backend="modal",
                        phase="notebook-execution",
                        diagnostics=stderr or stdout,
                    )
                sizes = await self._inventory_sizes()
                notebook_size = sizes.pop("notebook.ipynb", None)
                if notebook_size is None:
                    raise NotebookInfrastructureError(
                        "Modal execution did not return a notebook",
                        backend="modal",
                        phase="notebook-download",
                    )
                if notebook_size > self.limits.max_notebook_bytes:
                    raise NotebookLimitExceeded(
                        f"Notebook exceeds {self.limits.max_notebook_bytes} bytes",
                        backend="modal",
                        phase="notebook-download",
                    )
                executed = await _modal_operation(
                    self.sandbox.filesystem.read_bytes.aio(
                        f"{self.run_path}/notebook.ipynb"
                    ),
                    timeout=self.limits.open_seconds,
                    phase="notebook-download",
                )
            except NotebookExecutionTimeout:
                await self._terminate_after_interruption()
                raise
            except (
                NotebookInfrastructureError,
                NotebookImageError,
                NotebookLimitExceeded,
            ):
                raise
            except asyncio.CancelledError:
                await self._terminate_after_interruption()
                raise
            except TimeoutError as exc:
                await self._terminate_after_interruption()
                raise NotebookExecutionTimeout(
                    "Modal notebook execution timed out",
                    backend="modal",
                    phase="notebook-execution",
                ) from exc
            except Exception as exc:
                raise NotebookInfrastructureError(
                    "Could not transfer Modal notebook results",
                    backend="modal",
                    phase="notebook-download",
                    diagnostics=bound_log(str(exc), self.limits.max_log_chars),
                ) from exc

            validate_notebook_bytes(
                executed,
                self.limits,
                backend="modal",
                phase="notebook-download",
            )
            inventory = build_artifact_inventory(
                sizes,
                self.limits,
                backend="modal",
            )
            self._inventory = inventory
            return NotebookExecutionResult(executed, inventory, stdout, stderr)

    async def read_artifact(self, artifact: ArtifactInfo) -> bytes:
        async with self._lock:
            self._ensure_open()
            if artifact not in self._inventory:
                raise NotebookInfrastructureError(
                    f"Unknown artifact: {artifact.path}",
                    backend="modal",
                    phase="artifact-download",
                )
            relative = validate_artifact_path(artifact.path, backend="modal")
            try:
                data = await _modal_operation(
                    self.sandbox.filesystem.read_bytes.aio(
                        f"{self.run_path}/{relative.as_posix()}"
                    ),
                    timeout=self.limits.open_seconds,
                    phase="artifact-download",
                )
            except NotebookExecutionTimeout:
                await self._terminate_after_interruption()
                raise
            except Exception as exc:
                raise NotebookInfrastructureError(
                    f"Could not read artifact: {artifact.path}",
                    backend="modal",
                    phase="artifact-download",
                    diagnostics=bound_log(str(exc), self.limits.max_log_chars),
                ) from exc
            if len(data) != artifact.size or len(data) > self.limits.max_artifact_bytes:
                raise NotebookInfrastructureError(
                    f"Artifact changed after inventory: {artifact.path}",
                    backend="modal",
                    phase="artifact-download",
                )
            return bytes(data)

    async def list_workspace(self) -> tuple[ArtifactInfo, ...]:
        async with self._lock:
            self._ensure_open()
            return self._inventory

    async def close(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            self._inventory = ()
            await self._terminate()

    async def _preflight_runtime(self) -> None:
        returncode, stdout, stderr = await self._exec(
            "image-preflight",
            _RUNUSER,
            "-u",
            _RUN_USER,
            "--",
            "jupyter",
            "nbconvert",
            "--version",
            timeout=self.limits.open_seconds,
        )
        if returncode != 0:
            raise NotebookImageError(
                "Notebook image must provide jovyan, runuser, jupyter, and nbconvert",
                backend="modal",
                phase="image-preflight",
                diagnostics=stderr or stdout,
            )

    async def _inventory_sizes(self) -> dict[str, int]:
        returncode, stdout, stderr = await self._exec(
            "artifact-inventory",
            _PYTHON,
            "-c",
            _INVENTORY_SCRIPT,
            self.run_path,
            str(self.limits.max_artifacts),
            str(self.limits.max_artifact_bytes),
            str(self.limits.max_total_artifact_bytes),
            str(self.limits.max_notebook_bytes),
            timeout=self.limits.open_seconds,
        )
        if returncode != 0:
            raise NotebookLimitExceeded(
                "Modal artifact inventory exceeded limits or contained unsafe files",
                backend="modal",
                phase="artifact-inventory",
                diagnostics=stderr or stdout,
            )
        try:
            payload = json.loads(stdout)
            files = payload["files"]
            if not isinstance(files, dict):
                raise TypeError("files must be an object")
            return {str(path): int(size) for path, size in files.items()}
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise NotebookInfrastructureError(
                "Modal returned an invalid artifact inventory",
                backend="modal",
                phase="artifact-inventory",
                diagnostics=stdout,
            ) from exc

    async def _exec(
        self,
        phase: str,
        *argv: str,
        timeout: int | None,
        workdir: str | None = None,
    ) -> tuple[int, str, str]:
        try:
            async with asyncio.timeout(timeout):
                process = await self.sandbox.exec.aio(
                    *argv,
                    timeout=timeout,
                    workdir=workdir,
                )
                stdout, stderr, returncode = await asyncio.gather(
                    process.stdout.read.aio(),
                    process.stderr.read.aio(),
                    process.wait.aio(),
                )
        except TimeoutError as exc:
            raise NotebookExecutionTimeout(
                f"Modal {phase} timed out after {timeout} seconds",
                backend="modal",
                phase=phase,
            ) from exc
        return (
            returncode,
            bound_log(stdout, self.limits.max_log_chars),
            bound_log(stderr, self.limits.max_log_chars),
        )

    async def _terminate_after_interruption(self) -> None:
        self._closed = True
        task = asyncio.create_task(self._terminate())
        with contextlib.suppress(Exception):
            await asyncio.shield(task)

    async def _terminate(self) -> None:
        with contextlib.suppress(Exception):
            async with asyncio.timeout(30):
                await self.sandbox.terminate.aio(wait=True)

    def _ensure_open(self) -> None:
        if self._closed:
            raise NotebookInfrastructureError(
                "Notebook environment is closed",
                backend="modal",
                phase="lifecycle",
            )


def _bounded_timeout(
    requested: int | None,
    configured: int | None,
) -> int | None:
    if requested is None:
        return configured
    if configured is None:
        return requested
    return min(requested, configured)


async def _modal_operation(
    operation: Any,
    *,
    timeout: int,
    phase: str,
) -> Any:
    try:
        async with asyncio.timeout(timeout):
            return await operation
    except TimeoutError as exc:
        raise NotebookExecutionTimeout(
            f"Modal {phase} timed out after {timeout} seconds",
            backend="modal",
            phase=phase,
        ) from exc


def _load_modal() -> Any:
    try:
        import modal
    except ImportError as exc:
        raise NotebookBackendUnavailable(
            "Modal backend requires the `sqlsaber-notebook[modal]` extra",
            backend="modal",
            phase="availability",
        ) from exc
    return modal
