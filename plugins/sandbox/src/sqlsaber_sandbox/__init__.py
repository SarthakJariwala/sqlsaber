"""SQLSaber sandbox capability plugin."""

from typing import Any

from .backends import CommandResult, SandboxBackend, SandboxError, SessionLost
from .capability import Sandbox, capability
from .config import DEFAULT_SANDBOX_IMAGE, SandboxConfig, WorkspaceLimits
from .result import AnalysisResult, ArtifactRef, CellResult, Workspace, WorkspaceFile

_LAZY_EXPORTS = {
    "SandboxSession": ("session", "SandboxSession"),
    "publish_analysis": ("publication", "publish_analysis"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(name)
    from importlib import import_module

    module, attribute = _LAZY_EXPORTS[name]
    value = getattr(import_module(f"sqlsaber_sandbox.{module}"), attribute)
    globals()[name] = value
    return value


__all__ = [
    "Sandbox",
    "SandboxConfig",
    "DEFAULT_SANDBOX_IMAGE",
    "SandboxSession",
    "SandboxBackend",
    "CommandResult",
    "SandboxError",
    "SessionLost",
    "WorkspaceLimits",
    "Workspace",
    "WorkspaceFile",
    "AnalysisResult",
    "ArtifactRef",
    "CellResult",
    "capability",
    "publish_analysis",
]
