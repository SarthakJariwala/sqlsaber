"""Host-selected admission and execution budgets for sandbox analyses."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields

from sqlsaber.nested_model import INHERIT, NestedModel

from .result import Workspace, WorkspaceFile, validate_workspace_file_name

MIB = 1024 * 1024
DEFAULT_SANDBOX_IMAGE = (
    "quay.io/jupyter/scipy-notebook@sha256:"
    "e6e8baae46b5e62bbc26910169082639a6fd96f90e9f6fc52e0c0389df92d35c"
)
ANALYZE_IN_SANDBOX = "analyze_in_sandbox"


def _positive_integer(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _positive_number(name: str, value: object) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a finite positive number")


@dataclass(frozen=True, slots=True)
class WorkspaceLimits:
    """Admission budgets for files and their generated workspace manifest."""

    max_files: int = 50
    max_file_bytes: int = 100 * MIB
    max_total_bytes: int = 250 * MIB
    max_manifest_bytes: int = MIB
    default_results: int = 20

    def __post_init__(self) -> None:
        for item in fields(self):
            _positive_integer(item.name, getattr(self, item.name))

    def validate(self, files: Sequence[WorkspaceFile]) -> None:
        """Validate one complete user-file workspace before provisioning."""

        admitted = tuple(files)
        if len(admitted) > self.max_files:
            raise ValueError(
                f"Workspace has {len(admitted)} files; maximum is {self.max_files}"
            )

        names: set[str] = set()
        total_bytes = 0
        for item in admitted:
            if not isinstance(item, WorkspaceFile):
                raise ValueError("Workspace contains an invalid file")
            if item.name == "manifest.json":
                raise ValueError("Workspace filename 'manifest.json' is reserved")
            validate_workspace_file_name(item.name)
            if item.name in names:
                raise ValueError(f"Duplicate workspace filename: {item.name}")
            if len(item.data) > self.max_file_bytes:
                raise ValueError(
                    f"Workspace file exceeds {self.max_file_bytes} bytes: {item.name}"
                )
            names.add(item.name)
            total_bytes += len(item.data)

        if total_bytes > self.max_total_bytes:
            raise ValueError(f"Workspace exceeds {self.max_total_bytes} total bytes")

        manifest = Workspace(admitted).manifest_bytes()
        if len(manifest) > self.max_manifest_bytes:
            raise ValueError(
                f"Workspace manifest exceeds {self.max_manifest_bytes} bytes"
            )


@dataclass(frozen=True, slots=True)
class SandboxConfig:
    """Provider selection and host-enforced sandbox analysis budgets."""

    provider: str | None = None
    image: str | None = None
    cpu_cores: float | None = None
    memory_mb: int | None = None
    gpu: str | None = None
    workspace: WorkspaceLimits = WorkspaceLimits()
    open_seconds: int = 180
    transport_seconds: int = 30
    cell_seconds: float | None = 600
    idle_seconds: float | None = None
    max_lifetime_seconds: int | None = None
    max_artifacts: int = 50
    max_artifact_bytes: int = 50 * MIB
    max_total_artifact_bytes: int = 200 * MIB
    max_output_chars: int = 16_000
    max_image_bytes: int = 4 * MIB
    max_history_image_bytes: int = 24 * MIB
    model: NestedModel = INHERIT

    def __post_init__(self) -> None:
        if not isinstance(self.workspace, WorkspaceLimits):
            raise ValueError("workspace must be a WorkspaceLimits value")

        for name in ("provider", "image", "gpu"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{name} must be a string or None")
            if isinstance(value, str) and not value.strip():
                raise ValueError(f"{name} cannot be empty")

        if self.cpu_cores is not None:
            _positive_number("cpu_cores", self.cpu_cores)
        if self.memory_mb is not None:
            _positive_integer("memory_mb", self.memory_mb)

        for name in ("open_seconds", "transport_seconds"):
            _positive_integer(name, getattr(self, name))
        for name in ("cell_seconds", "idle_seconds"):
            value = getattr(self, name)
            if value is not None:
                _positive_number(name, value)
        if self.max_lifetime_seconds is not None:
            _positive_integer("max_lifetime_seconds", self.max_lifetime_seconds)

        for name in (
            "max_artifacts",
            "max_artifact_bytes",
            "max_total_artifact_bytes",
            "max_output_chars",
            "max_image_bytes",
            "max_history_image_bytes",
        ):
            _positive_integer(name, getattr(self, name))

    def controller_config(self) -> dict[str, object]:
        """Budgets the guest kernel enforces. Host model pins stay on the host."""
        payload = asdict(self)
        del payload["model"]
        return payload


DEFAULT_SANDBOX_CONFIG = SandboxConfig()
