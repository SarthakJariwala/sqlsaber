"""Central registry for supported AI providers.

This module defines a single source of truth for providers used across the
codebase (CLI, config, agents). Update this file to add or modify providers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class ProviderSpec:
    """Specification for a provider."""

    key: str
    env_var: str
    supports_oauth: bool = False
    aliases: tuple[str, ...] = ()


# Ordered definition -> used for CLI display order
_PROVIDERS: List[ProviderSpec] = [
    ProviderSpec(
        key="anthropic",
        env_var="ANTHROPIC_API_KEY",
        supports_oauth=True,
        aliases=(),
    ),
    ProviderSpec(
        key="openai",
        env_var="OPENAI_API_KEY",
        aliases=(),
    ),
    ProviderSpec(
        key="google",
        env_var="GOOGLE_API_KEY",
        # Historically some model IDs start with "google-gla"; treat as alias
        aliases=("google-gla",),
    ),
    ProviderSpec(
        key="groq",
        env_var="GROQ_API_KEY",
        aliases=(),
    ),
    ProviderSpec(
        key="mistral",
        env_var="MISTRAL_API_KEY",
        aliases=(),
    ),
    ProviderSpec(
        key="cohere",
        env_var="COHERE_API_KEY",
        aliases=(),
    ),
    ProviderSpec(
        key="huggingface",
        env_var="HUGGINGFACE_API_KEY",
        aliases=(),
    ),
]


# Fast lookup maps
_BY_KEY: Dict[str, ProviderSpec] = {p.key: p for p in _PROVIDERS}
_ALIAS_TO_KEY: Dict[str, str] = {
    alias: p.key for p in _PROVIDERS for alias in p.aliases
}


def all_keys() -> List[str]:
    """Return provider keys in display order."""
    return [p.key for p in _PROVIDERS]


def env_var_name(key: str) -> str:
    """Return the expected environment variable for a provider.

    Falls back to a generic name if the provider is unknown.
    """
    spec = _BY_KEY.get(key)
    return spec.env_var if spec else "AI_API_KEY"


def supports_oauth(key: str) -> bool:
    """Return True if the provider supports OAuth in SQLsaber."""
    spec = _BY_KEY.get(key)
    return bool(spec and spec.supports_oauth)


def canonical(key_or_alias: str) -> Optional[str]:
    """Return the canonical provider key for a provider or alias.

    Returns None if not recognized.
    """
    if key_or_alias in _BY_KEY:
        return key_or_alias
    return _ALIAS_TO_KEY.get(key_or_alias)


def provider_from_model(model_name: str) -> Optional[str]:
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
