"""Host-selected notebook configuration shared by both analysis entry points."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, fields

from .execution import (
    ExecutionLimits,
    NotebookBackend,
    NotebookInput,
    NotebookLimitExceeded,
)
from .execution.base import MIB, validate_inputs

_EXECUTION_DEFAULTS = ExecutionLimits()


def _positive_integer(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class WorkspaceLimits:
    """Admission budgets for user files, excluding the generated manifest."""

    max_files: int = 50
    max_file_bytes: int = 100 * MIB
    max_total_bytes: int = 250 * MIB
    max_manifest_bytes: int = MIB
    default_results: int = 20

    def __post_init__(self) -> None:
        for item in fields(self):
            _positive_integer(item.name, getattr(self, item.name))

    def validate_files(self, inputs: Sequence[NotebookInput]) -> None:
        if any(item.name == "manifest.json" for item in inputs):
            raise NotebookLimitExceeded(
                "Workspace filename 'manifest.json' is reserved",
                backend="notebook",
                phase="input-validation",
            )
        validate_inputs(
            inputs,
            ExecutionLimits(
                max_input_files=self.max_files,
                max_input_file_bytes=self.max_file_bytes,
                max_total_input_bytes=self.max_total_bytes,
            ),
            backend="notebook",
        )

    def validate_manifest(self, manifest: bytes) -> None:
        if len(manifest) > self.max_manifest_bytes:
            raise NotebookLimitExceeded(
                f"Workspace manifest exceeds {self.max_manifest_bytes} bytes "
                f"(requires {len(manifest)}; configure workspace.max_manifest_bytes)",
                backend="notebook",
                phase="input-validation",
            )


@dataclass(frozen=True, slots=True)
class NotebookConfig:
    """Approved workspace, execution, and export budgets, not model-preview budgets.

    Unspecified backend/image selectors fall back to environment settings. A None
    execution timeout disables that SQLsaber timer, not provider lifetime limits.
    """

    workspace: WorkspaceLimits = WorkspaceLimits()
    backend: NotebookBackend | str | None = None
    image: str | None = None
    image_prepare_seconds: int = _EXECUTION_DEFAULTS.image_prepare_seconds
    open_seconds: int = _EXECUTION_DEFAULTS.open_seconds
    cell_seconds: int | None = _EXECUTION_DEFAULTS.cell_seconds
    command_seconds: int | None = _EXECUTION_DEFAULTS.command_seconds
    memory_mb: int = _EXECUTION_DEFAULTS.memory_mb
    cpu_cores: float = _EXECUTION_DEFAULTS.cpu_cores
    pids: int = _EXECUTION_DEFAULTS.pids
    max_notebook_bytes: int = _EXECUTION_DEFAULTS.max_notebook_bytes
    max_artifacts: int = _EXECUTION_DEFAULTS.max_artifacts
    max_artifact_bytes: int = _EXECUTION_DEFAULTS.max_artifact_bytes
    max_total_artifact_bytes: int = _EXECUTION_DEFAULTS.max_total_artifact_bytes
    max_log_chars: int = _EXECUTION_DEFAULTS.max_log_chars

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name in {"workspace", "backend", "image", "cpu_cores"}:
                continue
            if item.name in {"cell_seconds", "command_seconds"} and value is None:
                continue
            _positive_integer(item.name, value)
        if (
            isinstance(self.cpu_cores, bool)
            or not isinstance(self.cpu_cores, (int, float))
            or not math.isfinite(self.cpu_cores)
            or self.cpu_cores <= 0
        ):
            raise ValueError("cpu_cores must be a finite positive number")
        for name in ("backend", "image"):
            value = getattr(self, name)
            if isinstance(value, str) and not value.strip():
                raise ValueError(f"{name} cannot be empty")

    def execution_limits(self) -> ExecutionLimits:
        """Translate user-file budgets to backend staging budgets exactly once."""

        return ExecutionLimits(
            image_prepare_seconds=self.image_prepare_seconds,
            open_seconds=self.open_seconds,
            cell_seconds=self.cell_seconds,
            command_seconds=self.command_seconds,
            memory_mb=self.memory_mb,
            cpu_cores=self.cpu_cores,
            pids=self.pids,
            max_input_files=self.workspace.max_files + 1,
            max_input_file_bytes=max(
                self.workspace.max_file_bytes, self.workspace.max_manifest_bytes
            ),
            max_total_input_bytes=(
                self.workspace.max_total_bytes + self.workspace.max_manifest_bytes
            ),
            max_notebook_bytes=self.max_notebook_bytes,
            max_artifacts=self.max_artifacts,
            max_artifact_bytes=self.max_artifact_bytes,
            max_total_artifact_bytes=self.max_total_artifact_bytes,
            max_log_chars=self.max_log_chars,
        )


DEFAULT_NOTEBOOK_CONFIG = NotebookConfig()
