"""Central registry for supported AI providers.

This module defines a single source of truth for providers used across the
codebase (CLI, config, agents). Update this file to add or modify providers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class AuthKind(str, Enum):
    """Credential mechanism used by a model provider."""

    API_KEY = "api-key"
    OPENAI_CODEX = "openai-codex"


@dataclass(frozen=True)
class ProviderSpec:
    """Specification for a provider."""

    key: str
    auth_kind: AuthKind
    env_var: str | None = None
    aliases: tuple[str, ...] = ()


_PROVIDERS: list[ProviderSpec] = [
    ProviderSpec(
        key="anthropic",
        auth_kind=AuthKind.API_KEY,
        env_var="ANTHROPIC_API_KEY",
        aliases=(),
    ),
    ProviderSpec(
        key="openai",
        auth_kind=AuthKind.API_KEY,
        env_var="OPENAI_API_KEY",
        aliases=(),
    ),
    ProviderSpec(
        key="openai-codex",
        auth_kind=AuthKind.OPENAI_CODEX,
    ),
    ProviderSpec(
        key="google",
        auth_kind=AuthKind.API_KEY,
        env_var="GOOGLE_API_KEY",
        aliases=("google-gla",),
    ),
    ProviderSpec(
        key="groq",
        auth_kind=AuthKind.API_KEY,
        env_var="GROQ_API_KEY",
        aliases=(),
    ),
    ProviderSpec(
        key="mistral",
        auth_kind=AuthKind.API_KEY,
        env_var="MISTRAL_API_KEY",
        aliases=(),
    ),
    ProviderSpec(
        key="cohere",
        auth_kind=AuthKind.API_KEY,
        env_var="COHERE_API_KEY",
        aliases=(),
    ),
    ProviderSpec(
        key="huggingface",
        auth_kind=AuthKind.API_KEY,
        env_var="HUGGINGFACE_API_KEY",
        aliases=(),
    ),
    ProviderSpec(
        key="xai",
        auth_kind=AuthKind.API_KEY,
        env_var="XAI_API_KEY",
        aliases=(),
    ),
]


_BY_KEY: dict[str, ProviderSpec] = {p.key: p for p in _PROVIDERS}
_ALIAS_TO_KEY: dict[str, str] = {
    alias: p.key for p in _PROVIDERS for alias in p.aliases
}


def all_keys() -> list[str]:
    """Return provider keys in display order."""
    return [p.key for p in _PROVIDERS]


def api_key_keys() -> list[str]:
    """Return providers that authenticate with API keys."""

    return [p.key for p in _PROVIDERS if p.auth_kind is AuthKind.API_KEY]


def auth_kind(key_or_alias: str) -> AuthKind | None:
    """Return the provider's credential mechanism."""

    key = canonical(key_or_alias)
    return _BY_KEY[key].auth_kind if key is not None else None


def env_var_name(key: str) -> str | None:
    """Return the expected environment variable for a provider.

    Falls back to a generic name if the provider is unknown.
    """
    spec = _BY_KEY.get(key)
    return spec.env_var if spec else "AI_API_KEY"


def canonical(key_or_alias: str) -> str | None:
    """Return the canonical provider key for a provider or alias.

    Returns None if not recognized.
    """
    if key_or_alias in _BY_KEY:
        return key_or_alias
    return _ALIAS_TO_KEY.get(key_or_alias)


def provider_from_model(model_name: str) -> str | None:
    """Infer the canonical provider key from a model identifier.

    Accepts either "provider:model_id" or a bare provider string. Aliases are
    normalized to their canonical provider key.
    """
    if not model_name:
        return None
    provider_raw = model_name.split(":", 1)[0]
    return canonical(provider_raw)


def specs() -> Iterable[ProviderSpec]:
    """Iterate provider specifications (in display order)."""
    return tuple(_PROVIDERS)
