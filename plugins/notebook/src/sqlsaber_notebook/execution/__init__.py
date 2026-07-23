"""Notebook-specific execution backends and selection."""

from __future__ import annotations

import os

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
)
from .docker import DockerNotebookBackend

DEFAULT_NOTEBOOK_BACKEND = "docker"
DEFAULT_NOTEBOOK_IMAGE = (
    "quay.io/jupyter/scipy-notebook@sha256:"
    "e6e8baae46b5e62bbc26910169082639a6fd96f90e9f6fc52e0c0389df92d35c"
)


def resolve_notebook_backend(name: str | None = None) -> NotebookBackend:
    """Resolve one explicitly selected backend without cross-domain fallback."""

    selected = name or os.getenv("SQLSABER_NOTEBOOK_BACKEND", DEFAULT_NOTEBOOK_BACKEND)
    normalized = selected.strip().lower()
    if normalized == "docker":
        return DockerNotebookBackend()
    if normalized == "microsandbox":
        from .microsandbox import MicrosandboxNotebookBackend

        return MicrosandboxNotebookBackend()
    if normalized == "modal":
        from .modal import ModalNotebookBackend

        return ModalNotebookBackend()
    raise NotebookBackendUnavailable(
        f"Unknown notebook backend {selected!r}; expected 'docker', "
        "'microsandbox', or 'modal'",
        backend=normalized or "unknown",
        phase="configuration",
    )


def resolve_notebook_image(image: str | None = None) -> str:
    """Resolve the immutable default image or an explicit environment override."""

    selected = image or os.getenv("SQLSABER_NOTEBOOK_IMAGE", DEFAULT_NOTEBOOK_IMAGE)
    if not selected.strip():
        raise NotebookImageError(
            "Notebook image cannot be empty",
            backend="configuration",
            phase="configuration",
        )
    return selected.strip()


__all__ = [
    "ArtifactInfo",
    "DEFAULT_NOTEBOOK_BACKEND",
    "DEFAULT_NOTEBOOK_IMAGE",
    "DockerNotebookBackend",
    "ExecutionLimits",
    "NotebookBackend",
    "NotebookBackendUnavailable",
    "NotebookEnvironment",
    "NotebookExecutionError",
    "NotebookExecutionResult",
    "NotebookExecutionTimeout",
    "NotebookImageError",
    "NotebookInfrastructureError",
    "NotebookInput",
    "NotebookLimitExceeded",
    "resolve_notebook_backend",
    "resolve_notebook_image",
]
