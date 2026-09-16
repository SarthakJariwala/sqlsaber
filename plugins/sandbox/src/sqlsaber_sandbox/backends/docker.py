"""Native local Docker backend owned through the Docker CLI."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Coroutine, Sequence
from dataclasses import dataclass
import shutil
from typing import Any
import uuid

from ..config import SandboxConfig
from .base import CommandResult, SandboxError


@dataclass(frozen=True, slots=True)
class _ProcessResult:
    exit_code: int
    stdout: bytes
    stderr: bytes


class DockerBackend:
    """One named container and its attached controller process."""

    def __init__(self, *, executable: str | None = None) -> None:
        self.executable = executable or shutil.which("docker") or "docker"
        self._config: SandboxConfig | None = None
        self._container_name: str | None = None
        self._controller: asyncio.subprocess.Process | None = None
        self._controller_drains: tuple[asyncio.Task[None], ...] = ()
        self._close_lock = asyncio.Lock()

    async def open(self, config: SandboxConfig) -> None:
        if self._container_name is not None:
            raise SandboxError("Docker sandbox is already open")
        if config.provider not in {None, "docker"}:
            raise ValueError(f"Docker backend cannot use provider {config.provider!r}")
        self._config = config
        try:
            info = await self._run(
                (self.executable, "info", "--format", "{{.ServerVersion}}")
            )
        except FileNotFoundError as exc:
            self._config = None
            raise SandboxError("Docker CLI is not installed") from exc
        if info.exit_code != 0:
            self._config = None
            raise SandboxError(
                f"Docker daemon is unavailable: {_diagnostic(info.stderr)}"
            )

        self._container_name = f"sqlsaber-sandbox-{uuid.uuid4().hex}"
        create_argv = [
            self.executable,
            "create",
            "--name",
            self._container_name,
            "--init",
            "--label",
            "sqlsaber.owner=sandbox",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
        ]
        if config.cpu_cores is not None:
            create_argv.extend(("--cpus", str(config.cpu_cores)))
        if config.memory_mb is not None:
            create_argv.extend(("--memory", f"{config.memory_mb}m"))
        if config.gpu is not None:
            create_argv.extend(("--gpus", config.gpu))
        create_argv.extend(
            (
                config.image or "python:3.12-slim",
                "python",
                "-c",
                "import time; time.sleep(2147483647)",
            )
        )

        try:
            created = await self._run(tuple(create_argv))
            if created.exit_code != 0:
                raise SandboxError(
                    f"Docker container creation failed: {_diagnostic(created.stderr)}"
                )
            started = await self._run((self.executable, "start", self._container_name))
            if started.exit_code != 0:
                raise SandboxError(
                    f"Docker container start failed: {_diagnostic(started.stderr)}"
                )
        except BaseException:
            await _preserve_primary(self.close())
            raise

    async def execute(self, command: str, *, timeout: float) -> CommandResult:
        container = self._require_container()
        try:
            result = await self._run(
                (self.executable, "exec", container, "sh", "-lc", command),
                timeout=timeout,
            )
        except asyncio.CancelledError:
            await _preserve_primary(self.close())
            raise
        except TimeoutError:
            # Docker exposes no native handle for an individual exec process.
            # Removing the owned container is the only way to guarantee that a
            # timed-out command is not still mutating persistent state.
            await _preserve_primary(self.close())
            raise
        return CommandResult(
            stdout=result.stdout.decode(errors="replace"),
            stderr=result.stderr.decode(errors="replace"),
            exit_code=result.exit_code,
        )

    async def upload(self, data: bytes, path: str) -> None:
        container = self._require_container()
        result = await self._run(
            (
                self.executable,
                "exec",
                "-i",
                container,
                "sh",
                "-c",
                'cat > "$1"',
                "sqlsaber-upload",
                path,
            ),
            input_data=data,
        )
        if result.exit_code != 0:
            raise SandboxError(f"Docker upload failed: {_diagnostic(result.stderr)}")

    async def download(self, path: str) -> bytes:
        container = self._require_container()
        result = await self._run(
            (
                self.executable,
                "exec",
                container,
                "sh",
                "-c",
                'cat "$1"',
                "sqlsaber-download",
                path,
            )
        )
        if result.exit_code != 0:
            raise SandboxError(f"Docker download failed: {_diagnostic(result.stderr)}")
        return result.stdout

    async def start_controller(self, argv: Sequence[str]) -> None:
        container = self._require_container()
        arguments = tuple(argv)
        if not arguments:
            raise ValueError("Controller argv cannot be empty")
        if self._controller is not None:
            raise SandboxError("Docker controller has already been started")
        try:
            process = await _spawn_process(
                (self.executable, "exec", container, *arguments),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise SandboxError("Docker CLI is not installed") from exc
        self._controller = process
        self._controller_drains = tuple(
            asyncio.create_task(_drain(stream))
            for stream in (process.stdout, process.stderr)
            if stream is not None
        )
        await asyncio.sleep(0)
        if process.returncode is not None:
            await self._stop_controller()
            raise SandboxError("Docker controller exited during startup")

    async def close(self) -> None:
        async with self._close_lock:
            controller_stopped = await self._stop_controller()
            container = self._container_name
            if container is None:
                if not controller_stopped:
                    raise SandboxError("Docker controller cleanup failed")
                self._config = None
                return
            timeout = (
                float(self._config.transport_seconds)
                if self._config is not None
                else 30.0
            )
            try:
                result = await self._run(
                    (self.executable, "rm", "-f", container), timeout=timeout
                )
            except FileNotFoundError as exc:
                raise SandboxError("Docker CLI is not installed") from exc
            if result.exit_code != 0 and not _container_is_missing(result.stderr):
                raise SandboxError(
                    f"Docker container cleanup failed: {_diagnostic(result.stderr)}"
                )
            self._container_name = None
            if not controller_stopped and not await self._stop_controller():
                raise SandboxError("Docker controller cleanup failed")
            self._config = None

    def _require_container(self) -> str:
        if self._container_name is None:
            raise SandboxError("Docker sandbox is not open")
        return self._container_name

    async def _run(
        self,
        argv: Sequence[str],
        *,
        input_data: bytes | None = None,
        timeout: float | None = None,
    ) -> _ProcessResult:
        return await _run_process(
            argv,
            input_data=input_data,
            timeout=timeout,
        )

    async def _stop_controller(self) -> bool:
        process = self._controller
        drains = self._controller_drains
        if process is None:
            return True
        timeout = min(
            float(self._config.transport_seconds) if self._config is not None else 5.0,
            5.0,
        )
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=timeout)
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(process.wait(), timeout=timeout)
        if process.returncode is None:
            return False
        if drains:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*drains, return_exceptions=True), timeout=timeout
                )
            except TimeoutError:
                for task in drains:
                    task.cancel()
                await asyncio.gather(*drains, return_exceptions=True)
        self._controller = None
        self._controller_drains = ()
        return True


async def _spawn_process(
    argv: Sequence[str],
    **kwargs: Any,
) -> asyncio.subprocess.Process:
    return await asyncio.create_subprocess_exec(*argv, **kwargs)


async def _run_process(
    argv: Sequence[str],
    *,
    input_data: bytes | None = None,
    timeout: float | None = None,
) -> _ProcessResult:
    process = await _spawn_process(
        argv,
        stdin=(
            asyncio.subprocess.PIPE
            if input_data is not None
            else asyncio.subprocess.DEVNULL
        ),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        if timeout is None:
            stdout, stderr = await process.communicate(input_data)
        else:
            async with asyncio.timeout(timeout):
                stdout, stderr = await process.communicate(input_data)
    except BaseException:
        cleanup = asyncio.create_task(_kill_and_drain(process))
        with contextlib.suppress(BaseException):
            await asyncio.shield(cleanup)
        raise
    assert process.returncode is not None
    return _ProcessResult(process.returncode, stdout, stderr)


async def _kill_and_drain(process: asyncio.subprocess.Process) -> None:
    if process.returncode is None:
        with contextlib.suppress(ProcessLookupError):
            process.kill()
    with contextlib.suppress(Exception):
        await process.communicate()


async def _drain(stream: asyncio.StreamReader) -> None:
    while await stream.read(64 * 1024):
        pass


async def _preserve_primary(cleanup: Coroutine[Any, Any, None]) -> None:
    task = asyncio.create_task(cleanup)
    with contextlib.suppress(BaseException):
        await asyncio.shield(task)


def _diagnostic(data: bytes) -> str:
    return data.decode(errors="replace").strip() or "no diagnostics"


def _container_is_missing(stderr: bytes) -> bool:
    return b"no such container" in stderr.lower()
