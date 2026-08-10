"""SQLsaber notebook analysis plugin."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .execution import (
    DEFAULT_NOTEBOOK_BACKEND,
    DEFAULT_NOTEBOOK_IMAGE,
    ArtifactInfo,
    DockerNotebookBackend,
    ExecutionLimits,
    NotebookBackend,
    NotebookEnvironment,
    NotebookExecutionResult,
    NotebookInput,
    resolve_notebook_backend,
    resolve_notebook_image,
    resolve_notebook_snapshot,
)

if TYPE_CHECKING:
    from .analyst import analyze
    from .publication import publish_analysis
    from .result import AnalysisResult, ArtifactRef, ManifestEntry, Workspace

_LAZY_EXPORTS = {
    "AnalysisResult": ("sqlsaber_notebook.result", "AnalysisResult"),
    "ArtifactRef": ("sqlsaber_notebook.result", "ArtifactRef"),
    "ManifestEntry": ("sqlsaber_notebook.result", "ManifestEntry"),
    "Workspace": ("sqlsaber_notebook.result", "Workspace"),
    "analyze": ("sqlsaber_notebook.analyst", "analyze"),
    "publish_analysis": ("sqlsaber_notebook.publication", "publish_analysis"),
}


def __getattr__(name: str) -> Any:
    lazy_export = _LAZY_EXPORTS.get(name)
    if lazy_export is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module_name, attribute_name = lazy_export
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


__all__ = [
    "AnalysisResult",
    "ArtifactInfo",
    "ArtifactRef",
    "DEFAULT_NOTEBOOK_BACKEND",
    "DEFAULT_NOTEBOOK_IMAGE",
    "DockerNotebookBackend",
    "ExecutionLimits",
    "NotebookBackend",
    "NotebookEnvironment",
    "NotebookExecutionResult",
    "ManifestEntry",
    "NotebookInput",
    "Workspace",
    "analyze",
    "publish_analysis",
    "resolve_notebook_backend",
    "resolve_notebook_image",
    "resolve_notebook_snapshot",
]
