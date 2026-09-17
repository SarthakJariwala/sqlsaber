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


def create_backend(
    provider: str | None,
    *,
    e2b_api_key: str | None = None,
    daytona_api_key: str | None = None,
    daytona_api_url: str | None = None,
    modal_token_id: str | None = None,
    modal_token_secret: str | None = None,
    sprites_token: str | None = None,
) -> SandboxBackend:
    name = resolve_provider(provider)
    module = import_module(f"{__name__}.{name}")
    backend = getattr(module, _BACKENDS[name])
    if name == "e2b":
        return backend(api_key=e2b_api_key)
    if name == "daytona":
        return backend(api_key=daytona_api_key, api_url=daytona_api_url)
    if name == "modal":
        return backend(token_id=modal_token_id, token_secret=modal_token_secret)
    if name == "sprites":
        return backend(token=sprites_token)
    return backend()


__all__ = [
    "CommandResult",
    "SandboxBackend",
    "SandboxError",
    "SessionLost",
    "create_backend",
]
