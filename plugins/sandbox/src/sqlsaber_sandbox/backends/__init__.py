"""Lazy selection of SQLsaber-owned native sandbox backends."""

from importlib import import_module
import os

from .base import CommandResult, SandboxBackend, SandboxError, SessionLost

_BACKENDS = {
    "docker": "DockerBackend",
    "microsandbox": "MicrosandboxBackend",
    "modal": "ModalBackend",
    "e2b": "E2BBackend",
    "daytona": "DaytonaBackend",
    "sprites": "SpritesBackend",
}


def resolve_provider(provider: str | None) -> str:
    provider = provider or os.getenv("SQLSABER_SANDBOX_PROVIDER")
    if provider is not None:
        if provider not in _BACKENDS:
            raise ValueError(f"Unsupported sandbox provider: {provider}")
        return provider
    configured = [
        name
        for name, keys in (
            ("e2b", ("E2B_API_KEY",)),
            ("daytona", ("DAYTONA_API_KEY",)),
            ("modal", ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET")),
            ("sprites", ("SPRITES_TOKEN",)),
        )
        if all(os.getenv(key) for key in keys)
    ]
    if len(configured) != 1:
        raise ValueError(
            "Select SandboxConfig(provider=...) when zero or multiple providers are configured; "
            "local providers docker and microsandbox always require explicit selection"
        )
    return configured[0]


def create_backend(provider: str | None) -> SandboxBackend:
    name = resolve_provider(provider)
    module = import_module(f"{__name__}.{name}")
    return getattr(module, _BACKENDS[name])()


__all__ = [
    "CommandResult",
    "SandboxBackend",
    "SandboxError",
    "SessionLost",
    "create_backend",
]
