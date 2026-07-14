"""Provider-neutral notebook execution contracts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Protocol, runtime_checkable

MIB = 1024 * 1024


@dataclass(frozen=True, slots=True)
class ExecutionLimits:
    """Resource and transfer limits shared by every execution backend."""

    image_prepare_seconds: int = 600
    open_seconds: int = 180
    cell_seconds: int = 120
    command_seconds: int = 600
    operation_seconds: int = 1500
    idle_seconds: int = 300
    memory_mb: int = 4096
    cpu_cores: float = 2.0
    pids: int = 256
    # Twenty user-visible files plus the adapter-neutral manifest.json.
    max_input_files: int = 21
    max_input_file_bytes: int = 10 * MIB
    # User-visible quota is 50 MiB; reserve 1 MiB for manifest metadata.
    max_total_input_bytes: int = 51 * MIB
    max_notebook_bytes: int = 20 * MIB
    max_artifacts: int = 20
    max_artifact_bytes: int = 20 * MIB
    max_total_artifact_bytes: int = 50 * MIB
    max_log_chars: int = 8_000


@dataclass(frozen=True, slots=True)
class NotebookInput:
    """One immutable file staged for notebook code under ``../inputs``."""

    name: str
    data: bytes


@dataclass(frozen=True, slots=True)
class ArtifactInfo:
    """Metadata for a generated file in the latest clean run directory."""

    path: str
    size: int
    media_type: str | None = None


@dataclass(frozen=True, slots=True)
class NotebookExecutionResult:
    """Executed notebook plus a bounded inventory of generated files."""

    notebook: bytes
    artifacts: tuple[ArtifactInfo, ...]
    stdout: str = ""
    stderr: str = ""


class NotebookExecutionError(RuntimeError):
    """Base class for normalized backend failures."""

    def __init__(
        self,
        message: str,
        *,
        backend: str,
        phase: str,
        diagnostics: str = "",
    ) -> None:
        super().__init__(message)
        self.backend = backend
        self.phase = phase
        self.diagnostics = diagnostics


class NotebookBackendUnavailable(NotebookExecutionError):
    """The selected backend cannot be used in the current environment."""


class NotebookImageError(NotebookExecutionError):
    """The configured runtime image could not be prepared."""


class NotebookExecutionTimeout(NotebookExecutionError):
    """A bounded backend operation timed out."""


class NotebookInfrastructureError(NotebookExecutionError):
    """Execution, transfer, validation, or cleanup infrastructure failed."""


class NotebookLimitExceeded(NotebookExecutionError):
    """An input, notebook, artifact, or output exceeded a configured limit."""


@runtime_checkable
class NotebookEnvironment(Protocol):
    """One notebook-analysis environment with immutable staged inputs."""

    async def execute(
        self,
        notebook: bytes,
        *,
        cell_timeout: int,
        command_timeout: int,
    ) -> NotebookExecutionResult: ...

    async def read_artifact(self, artifact: ArtifactInfo) -> bytes: ...

    async def list_workspace(self) -> tuple[ArtifactInfo, ...]: ...

    async def close(self) -> None: ...


@runtime_checkable
class NotebookBackend(Protocol):
    """Factory for provider-specific notebook environments."""

    name: str

    def available(self) -> bool: ...

    async def open(
        self,
        inputs: Sequence[NotebookInput],
        *,
        image: str,
        limits: ExecutionLimits,
    ) -> NotebookEnvironment: ...


def validate_inputs(
    inputs: Sequence[NotebookInput],
    limits: ExecutionLimits,
    *,
    backend: str,
) -> tuple[NotebookInput, ...]:
    """Validate and freeze an input collection before opening an environment."""

    if len(inputs) > limits.max_input_files:
        raise NotebookLimitExceeded(
            f"Workspace has {len(inputs)} files; maximum is {limits.max_input_files}",
            backend=backend,
            phase="input-validation",
        )

    validated: list[NotebookInput] = []
    names: set[str] = set()
    total = 0
    for item in inputs:
        if not isinstance(item.data, bytes):
            raise TypeError("NotebookInput.data must be bytes")
        validate_input_name(item.name, backend=backend)
        if item.name in names:
            raise NotebookLimitExceeded(
                f"Duplicate workspace filename: {item.name}",
                backend=backend,
                phase="input-validation",
            )
        if len(item.data) > limits.max_input_file_bytes:
            raise NotebookLimitExceeded(
                f"Workspace file exceeds {limits.max_input_file_bytes} bytes: {item.name}",
                backend=backend,
                phase="input-validation",
            )
        names.add(item.name)
        total += len(item.data)
        validated.append(NotebookInput(item.name, bytes(item.data)))

    if total > limits.max_total_input_bytes:
        raise NotebookLimitExceeded(
            f"Workspace exceeds {limits.max_total_input_bytes} total bytes",
            backend=backend,
            phase="input-validation",
        )
    return tuple(validated)


def validate_input_name(name: str, *, backend: str) -> None:
    """Require a visible, single-component POSIX workspace filename."""

    path = PurePosixPath(name)
    if (
        not name
        or name in {".", ".."}
        or path.name != name
        or path.is_absolute()
        or name.startswith(".")
        or "\\" in name
        or "\x00" in name
    ):
        raise NotebookLimitExceeded(
            f"Unsafe workspace filename: {name!r}",
            backend=backend,
            phase="input-validation",
        )


def validate_artifact_path(path: str, *, backend: str) -> PurePosixPath:
    """Validate a generated path relative to the clean run directory."""

    candidate = PurePosixPath(path)
    if (
        not path
        or candidate.is_absolute()
        or any(
            part in {"", ".", ".."} or part.startswith(".") for part in candidate.parts
        )
        or "\\" in path
        or "\x00" in path
    ):
        raise NotebookInfrastructureError(
            f"Backend returned an unsafe artifact path: {path!r}",
            backend=backend,
            phase="artifact-inventory",
        )
    return candidate


def bound_log(value: bytes | str, limit: int) -> str:
    """Decode and retain both ends of backend diagnostics."""

    text = (
        value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
    )
    text = "".join(char for char in text if char in "\n\r\t" or ord(char) >= 32)
    if len(text) <= limit:
        return text
    marker = "\n...[backend log truncated]...\n"
    remaining = max(0, limit - len(marker))
    head = remaining // 2
    return f"{text[:head]}{marker}{text[-(remaining - head) :]}"
