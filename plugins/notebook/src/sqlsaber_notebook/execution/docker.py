"""Hardened local Docker implementation of notebook execution."""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import stat
import tempfile
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from ._files import build_artifact_inventory, validate_notebook_bytes
from .base import (
    ArtifactInfo,
    ExecutionLimits,
    NotebookBackend,
    NotebookBackendUnavailable,
    NotebookEnvironment,
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

Cleanup = Callable[[], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class _ProcessResult:
    returncode: int
    stdout: bytes
    stderr: bytes


class DockerNotebookBackend(NotebookBackend):
    """Open notebook environments backed by the local Docker CLI."""

    name = "docker"

    def __init__(self, *, executable: str | None = None) -> None:
        self.executable = executable or shutil.which("docker") or "docker"

    def available(self) -> bool:
        return (
            shutil.which(self.executable) is not None
            if os.path.sep not in self.executable
            else Path(self.executable).is_file()
        )

    async def open(
        self,
        inputs: Sequence[NotebookInput],
        *,
        image: str,
        limits: ExecutionLimits,
    ) -> DockerNotebookEnvironment:
        if not self.available():
            raise NotebookBackendUnavailable(
                "Docker CLI is not installed; install/start Docker or explicitly select Modal",
                backend=self.name,
                phase="availability",
            )
        validated = validate_inputs(inputs, limits, backend=self.name)
        await self._check_daemon(limits)
        await self._prepare_image(image, limits)
        root: Path | None = None
        try:
            root = await asyncio.to_thread(
                lambda: Path(tempfile.mkdtemp(prefix="sqlsaber-notebook-"))
            )
            await asyncio.to_thread(_stage_inputs, root, validated)
        except OSError as exc:
            if root is not None:
                await asyncio.to_thread(shutil.rmtree, root, True)
            raise NotebookInfrastructureError(
                "Could not create the private Docker notebook workspace",
                backend=self.name,
                phase="input-upload",
            ) from exc
        return DockerNotebookEnvironment(
            executable=self.executable,
            root=root,
            image=image,
            limits=limits,
        )

    async def _check_daemon(self, limits: ExecutionLimits) -> None:
        try:
            result = await _run_process(
                (self.executable, "info", "--format", "{{.ServerVersion}}"),
                timeout=limits.open_seconds,
                backend=self.name,
                phase="availability",
                log_limit=limits.max_log_chars,
            )
        except FileNotFoundError as exc:
            raise NotebookBackendUnavailable(
                "Docker CLI is not installed",
                backend=self.name,
                phase="availability",
            ) from exc
        if result.returncode != 0:
            raise NotebookBackendUnavailable(
                "Docker daemon is unavailable; start Docker and retry",
                backend=self.name,
                phase="availability",
                diagnostics=bound_log(result.stderr, limits.max_log_chars),
            )

    async def _prepare_image(self, image: str, limits: ExecutionLimits) -> None:
        inspect_result = await _run_process(
            (self.executable, "image", "inspect", image),
            timeout=limits.open_seconds,
            backend=self.name,
            phase="image-inspect",
            log_limit=limits.max_log_chars,
        )
        if inspect_result.returncode == 0:
            return
        pull_result = await _run_process(
            (self.executable, "pull", image),
            timeout=limits.image_prepare_seconds,
            backend=self.name,
            phase="image-pull",
            log_limit=limits.max_log_chars,
        )
        if pull_result.returncode != 0:
            raise NotebookImageError(
                f"Could not pull notebook image {image!r}",
                backend=self.name,
                phase="image-pull",
                diagnostics=bound_log(pull_result.stderr, limits.max_log_chars),
            )


class DockerNotebookEnvironment(NotebookEnvironment):
    """Private staged workspace that runs one disposable container per edit."""

    def __init__(
        self,
        *,
        executable: str,
        root: Path,
        image: str,
        limits: ExecutionLimits,
    ) -> None:
        self.executable = executable
        self.root = root
        self.inputs_path = root / "inputs"
        self.run_path = root / "run"
        self.image = image
        self.limits = limits
        self._inventory: tuple[ArtifactInfo, ...] = ()
        self._active_containers: set[str] = set()
        self._lock = asyncio.Lock()
        self._closed = False

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
                backend="docker",
                phase="notebook-upload",
            )
            await asyncio.to_thread(_reset_run_directory, self.run_path, notebook)
            container_name = f"sqlsaber-notebook-{uuid.uuid4().hex}"
            self._active_containers.add(container_name)
            try:
                argv = self._docker_argv(container_name, cell_timeout)
                result = await _run_process(
                    argv,
                    timeout=_bounded_timeout(
                        command_timeout, self.limits.command_seconds
                    ),
                    backend="docker",
                    phase="notebook-execution",
                    log_limit=self.limits.max_log_chars,
                    cleanup=lambda: self._force_remove(container_name),
                )
            finally:
                self._active_containers.discard(container_name)
            if result.returncode != 0:
                raise NotebookInfrastructureError(
                    "Docker notebook execution failed",
                    backend="docker",
                    phase="notebook-execution",
                    diagnostics=bound_log(
                        result.stderr or result.stdout,
                        self.limits.max_log_chars,
                    ),
                )

            try:
                sizes = await asyncio.to_thread(
                    _scan_artifact_sizes, self.run_path, self.limits
                )
                notebook_size = sizes.pop("notebook.ipynb", None)
                if notebook_size is None:
                    raise ValueError("execution did not return notebook.ipynb")
                if notebook_size > self.limits.max_notebook_bytes:
                    raise NotebookLimitExceeded(
                        f"Notebook exceeds {self.limits.max_notebook_bytes} bytes",
                        backend="docker",
                        phase="notebook-download",
                    )
                executed = await asyncio.to_thread(
                    (self.run_path / "notebook.ipynb").read_bytes
                )
            except (OSError, ValueError) as exc:
                raise NotebookInfrastructureError(
                    "Could not inspect Docker notebook results",
                    backend="docker",
                    phase="notebook-download",
                ) from exc
            validate_notebook_bytes(
                executed,
                self.limits,
                backend="docker",
                phase="notebook-download",
            )
            inventory = build_artifact_inventory(
                sizes,
                self.limits,
                backend="docker",
            )
            self._inventory = inventory
            return NotebookExecutionResult(
                notebook=executed,
                artifacts=inventory,
                stdout=bound_log(result.stdout, self.limits.max_log_chars),
                stderr=bound_log(result.stderr, self.limits.max_log_chars),
            )

    async def read_artifact(self, artifact: ArtifactInfo) -> bytes:
        async with self._lock:
            self._ensure_open()
            if artifact not in self._inventory:
                raise NotebookInfrastructureError(
                    f"Unknown artifact: {artifact.path}",
                    backend="docker",
                    phase="artifact-download",
                )
            relative = validate_artifact_path(artifact.path, backend="docker")
            path = self.run_path.joinpath(*relative.parts)
            try:
                metadata = await asyncio.to_thread(path.lstat)
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_size != artifact.size
                ):
                    raise ValueError("artifact changed after inventory")
                data = await asyncio.to_thread(path.read_bytes)
            except (OSError, ValueError) as exc:
                raise NotebookInfrastructureError(
                    f"Could not read artifact: {artifact.path}",
                    backend="docker",
                    phase="artifact-download",
                ) from exc
            if len(data) > self.limits.max_artifact_bytes:
                raise NotebookLimitExceeded(
                    f"Artifact exceeds {self.limits.max_artifact_bytes} bytes: {artifact.path}",
                    backend="docker",
                    phase="artifact-download",
                )
            return data

    async def list_workspace(self) -> tuple[ArtifactInfo, ...]:
        async with self._lock:
            self._ensure_open()
            return self._inventory

    async def close(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            containers = tuple(self._active_containers)
            self._active_containers.clear()
            for name in containers:
                with contextlib.suppress(Exception):
                    await self._force_remove(name)
            await asyncio.to_thread(shutil.rmtree, self.root, True)
            self._inventory = ()

    def _docker_argv(
        self, container_name: str, cell_timeout: int | None
    ) -> tuple[str, ...]:
        argv = [
            self.executable,
            "run",
            "--rm",
            "--name",
            container_name,
            "--network",
            "none",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            str(self.limits.pids),
            "--memory",
            f"{self.limits.memory_mb}m",
            "--cpus",
            str(self.limits.cpu_cores),
        ]
        if hasattr(os, "getuid") and hasattr(os, "getgid"):
            argv.extend(
                ("--user", f"{os.getuid()}:{os.getgid()}", "--env", "HOME=/tmp")
            )
        argv.extend(
            (
                "--mount",
                f"type=bind,source={self.inputs_path},target=/work/inputs,readonly",
                "--mount",
                f"type=bind,source={self.run_path},target=/work/run",
                "--workdir",
                "/work/run",
                self.image,
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
            )
        )
        return tuple(argv)

    async def _force_remove(self, container_name: str) -> None:
        with contextlib.suppress(Exception):
            await _run_process(
                (self.executable, "rm", "-f", container_name),
                timeout=10,
                backend="docker",
                phase="cleanup",
                log_limit=self.limits.max_log_chars,
            )

    def _ensure_open(self) -> None:
        if self._closed:
            raise NotebookInfrastructureError(
                "Notebook environment is closed",
                backend="docker",
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


async def _run_process(
    argv: Sequence[str],
    *,
    timeout: int | float | None,
    backend: str,
    phase: str,
    log_limit: int,
    cleanup: Cleanup | None = None,
) -> _ProcessResult:
    process = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        async with asyncio.timeout(timeout):
            stdout, stderr, _ = await asyncio.gather(
                _read_bounded_stream(process.stdout, log_limit),
                _read_bounded_stream(process.stderr, log_limit),
                process.wait(),
            )
    except TimeoutError as exc:
        await _stop_process(process, cleanup)
        raise NotebookExecutionTimeout(
            f"{phase} timed out after {timeout} seconds",
            backend=backend,
            phase=phase,
        ) from exc
    except asyncio.CancelledError:
        await _stop_process(process, cleanup)
        raise
    return _ProcessResult(
        process.returncode or 0,
        stdout,
        stderr,
    )


async def _read_bounded_stream(
    stream: asyncio.StreamReader | None,
    log_limit: int,
) -> bytes:
    if stream is None:
        return b""
    byte_limit = max(log_limit * 4, 32_000)
    head_limit = byte_limit // 2
    tail_limit = byte_limit - head_limit
    head = bytearray()
    tail = bytearray()
    while chunk := await stream.read(64 * 1024):
        if len(head) < head_limit:
            consumed = min(head_limit - len(head), len(chunk))
            head.extend(chunk[:consumed])
            chunk = chunk[consumed:]
        if chunk:
            tail.extend(chunk)
            if len(tail) > tail_limit:
                del tail[:-tail_limit]
    return bytes(head + tail)


async def _stop_process(
    process: asyncio.subprocess.Process, cleanup: Cleanup | None
) -> None:
    if process.returncode is None:
        with contextlib.suppress(ProcessLookupError):
            process.kill()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(process.wait(), timeout=5)
    if cleanup is not None:
        cleanup_task = asyncio.ensure_future(cleanup())
        with contextlib.suppress(Exception):
            await asyncio.shield(cleanup_task)


def _stage_inputs(root: Path, inputs: Sequence[NotebookInput]) -> None:
    inputs_path = root / "inputs"
    run_path = root / "run"
    inputs_path.mkdir(mode=0o700)
    run_path.mkdir(mode=0o700)
    for item in inputs:
        path = inputs_path / item.name
        path.write_bytes(item.data)
        path.chmod(0o444)


def _reset_run_directory(run_path: Path, notebook: bytes) -> None:
    shutil.rmtree(run_path, ignore_errors=True)
    run_path.mkdir(mode=0o700)
    (run_path / "notebook.ipynb").write_bytes(notebook)


def _scan_artifact_sizes(
    run_path: Path,
    limits: ExecutionLimits | None = None,
) -> dict[str, int]:
    limits = limits or ExecutionLimits()
    sizes: dict[str, int] = {}
    total = 0
    for root, directory_names, file_names in os.walk(run_path, followlinks=False):
        root_path = Path(root)
        for name in tuple(directory_names):
            path = root_path / name
            if path.is_symlink():
                raise ValueError(f"symlinked artifact directory: {path}")
        for name in file_names:
            path = root_path / name
            metadata = path.lstat()
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError(f"non-regular artifact: {path}")
            relative = path.relative_to(run_path).as_posix()
            sizes[relative] = metadata.st_size
            total += metadata.st_size
            if len(sizes) > limits.max_artifacts + 1:
                raise NotebookLimitExceeded(
                    "Run produced too many files",
                    backend="docker",
                    phase="artifact-inventory",
                )
            if metadata.st_size > max(
                limits.max_artifact_bytes, limits.max_notebook_bytes
            ):
                raise NotebookLimitExceeded(
                    f"Generated file is too large: {relative}",
                    backend="docker",
                    phase="artifact-inventory",
                )
            if total > limits.max_total_artifact_bytes + limits.max_notebook_bytes:
                raise NotebookLimitExceeded(
                    "Run produced too many artifact bytes",
                    backend="docker",
                    phase="artifact-inventory",
                )
    return sizes
