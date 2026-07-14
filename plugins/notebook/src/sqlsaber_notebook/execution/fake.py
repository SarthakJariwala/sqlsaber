"""Deterministic in-memory execution backend for core and contract tests."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field

from ._files import build_artifact_inventory, validate_notebook_bytes
from .base import (
    ArtifactInfo,
    ExecutionLimits,
    NotebookBackend,
    NotebookEnvironment,
    NotebookExecutionResult,
    NotebookInfrastructureError,
    NotebookInput,
    validate_inputs,
)


@dataclass(frozen=True, slots=True)
class FakeRunResult:
    notebook: bytes
    artifacts: Mapping[str, bytes] = field(default_factory=dict)
    stdout: str = ""
    stderr: str = ""


FakeExecutor = Callable[
    [bytes, Mapping[str, bytes], int],
    FakeRunResult | Awaitable[FakeRunResult],
]


class FakeNotebookBackend(NotebookBackend):
    """Open isolated in-memory environments driven by a supplied executor."""

    name = "fake"

    def __init__(self, executor: FakeExecutor | None = None) -> None:
        self.executor = executor or _echo_executor
        self.environments: list[FakeNotebookEnvironment] = []

    def available(self) -> bool:
        return True

    async def open(
        self,
        inputs: Sequence[NotebookInput],
        *,
        image: str,
        limits: ExecutionLimits,
    ) -> FakeNotebookEnvironment:
        del image
        validated = validate_inputs(inputs, limits, backend=self.name)
        environment = FakeNotebookEnvironment(validated, limits, self.executor)
        self.environments.append(environment)
        return environment


class FakeNotebookEnvironment(NotebookEnvironment):
    def __init__(
        self,
        inputs: Sequence[NotebookInput],
        limits: ExecutionLimits,
        executor: FakeExecutor,
    ) -> None:
        self.inputs = {item.name: item.data for item in inputs}
        self.limits = limits
        self.executor = executor
        self.run_count = 0
        self.closed = False
        self._artifacts: dict[str, bytes] = {}
        self._inventory: tuple[ArtifactInfo, ...] = ()

    async def execute(
        self,
        notebook: bytes,
        *,
        cell_timeout: int,
        command_timeout: int,
    ) -> NotebookExecutionResult:
        del cell_timeout, command_timeout
        self._ensure_open()
        validate_notebook_bytes(
            notebook,
            self.limits,
            backend="fake",
            phase="notebook-upload",
        )
        self.run_count += 1
        execution = self.executor(notebook, dict(self.inputs), self.run_count)
        if isinstance(execution, FakeRunResult):
            outcome = execution
        else:
            outcome = await execution
        validate_notebook_bytes(
            outcome.notebook,
            self.limits,
            backend="fake",
            phase="notebook-download",
        )
        artifacts = {path: bytes(data) for path, data in outcome.artifacts.items()}
        inventory = build_artifact_inventory(
            {path: len(data) for path, data in artifacts.items()},
            self.limits,
            backend="fake",
        )
        included = {item.path for item in inventory}
        self._artifacts = {
            path: data for path, data in artifacts.items() if path in included
        }
        self._inventory = inventory
        return NotebookExecutionResult(
            notebook=bytes(outcome.notebook),
            artifacts=inventory,
            stdout=outcome.stdout,
            stderr=outcome.stderr,
        )

    async def read_artifact(self, artifact: ArtifactInfo) -> bytes:
        self._ensure_open()
        known = next((item for item in self._inventory if item == artifact), None)
        if known is None or artifact.path not in self._artifacts:
            raise NotebookInfrastructureError(
                f"Unknown artifact: {artifact.path}",
                backend="fake",
                phase="artifact-download",
            )
        return bytes(self._artifacts[artifact.path])

    async def list_workspace(self) -> tuple[ArtifactInfo, ...]:
        self._ensure_open()
        return self._inventory

    async def close(self) -> None:
        self.closed = True
        self._artifacts.clear()
        self._inventory = ()

    def _ensure_open(self) -> None:
        if self.closed:
            raise NotebookInfrastructureError(
                "Notebook environment is closed",
                backend="fake",
                phase="lifecycle",
            )


def _echo_executor(
    notebook: bytes,
    inputs: Mapping[str, bytes],
    run_count: int,
) -> FakeRunResult:
    del inputs, run_count
    return FakeRunResult(notebook)
