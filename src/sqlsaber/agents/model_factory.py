"""Minimal pydantic-ai model construction helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

from pydantic_ai.models import Model
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.openai import OpenAIProvider

from sqlsaber.config import providers
from sqlsaber.config.settings import ThinkingLevel

if TYPE_CHECKING:
    from pydantic_ai.providers.openai_codex import OpenAICodexCredentialSource
    from sqlsaber.config.openai_codex import PreflightOpenAICodexCredentialSource
    from sqlsaber.nested_model import NestedModel

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

type UnifiedEffort = Literal["minimal", "low", "medium", "high", "xhigh"]

UNIFIED_EFFORT_MAP: dict[ThinkingLevel, UnifiedEffort] = {
    ThinkingLevel.MINIMAL: "minimal",
    ThinkingLevel.LOW: "low",
    ThinkingLevel.MEDIUM: "medium",
    ThinkingLevel.HIGH: "high",
    ThinkingLevel.MAXIMUM: "xhigh",
}


class ModelAuth(Protocol):
    """Authentication operations needed during model resolution."""

    def get_api_key(self, model_name: str) -> str | None: ...
    def validate(self, model_name: str) -> None: ...


@dataclass(frozen=True, slots=True)
class ResolvedModel:
    """A model plus the provider and API key inherited by child agents."""

    model_name: str
    model: Model | str
    provider: str
    api_key: str | None


def build_model(
    full_model_str: str,
    api_key: str | None,
    *,
    codex_credential_source: OpenAICodexCredentialSource | None = None,
) -> Model | str:
    """Build a provider model only when explicit credentials require it.

    OpenAI intentionally uses the Responses API model. Without an explicit key,
    returning a provider-prefixed string lets pydantic-ai perform normal discovery.
    """
    provider = providers.provider_from_model(full_model_str)
    provider_prefix, separator, model_name = full_model_str.partition(":")
    if not separator:
        model_name = full_model_str

    # Normalize SQLSaber's provider aliases before pydantic-ai sees the string.
    normalized_model_str = full_model_str
    if provider is not None and provider_prefix != provider:
        normalized_model_str = f"{provider}:{model_name}"

    if provider == "openai-codex":
        if api_key:
            raise ValueError("OpenAI Codex subscription models do not accept API keys.")
        if codex_credential_source is None:
            raise ValueError("An explicit OpenAI Codex credential source is required.")
        from pydantic_ai.models.openai_codex import OpenAICodexModel
        from pydantic_ai.providers.openai_codex import OpenAICodexProvider

        return OpenAICodexModel(
            model_name,
            provider=OpenAICodexProvider(
                credential_source=codex_credential_source,
            ),
        )

    if not api_key:
        return normalized_model_str
    if provider == "anthropic":
        return AnthropicModel(
            model_name,
            provider=AnthropicProvider(api_key=api_key),
        )
    if provider == "google":
        return GoogleModel(model_name, provider=GoogleProvider(api_key=api_key))
    if provider == "openai":
        return OpenAIResponsesModel(
            model_name,
            provider=OpenAIProvider(api_key=api_key),
        )
    if provider == "xai":
        from pydantic_ai.models.xai import XaiModel
        from pydantic_ai.providers.xai import XaiProvider

        return XaiModel(model_name, provider=XaiProvider(api_key=api_key))
    return normalized_model_str


def resolve_model(
    auth: ModelAuth,
    full_model_str: str,
    *,
    api_key_override: str | None = None,
    codex_credential_source: PreflightOpenAICodexCredentialSource | None = None,
) -> ResolvedModel:
    """Resolve authentication and construct one Pydantic AI model."""

    provider = providers.provider_from_model(full_model_str)
    if provider is None:
        provider = full_model_str.partition(":")[0].strip().lower()

    if provider == "openai-codex":
        if api_key_override:
            raise ValueError("OpenAI Codex subscription models do not accept API keys.")
        if codex_credential_source is None:
            from sqlsaber.config.openai_codex import OpenAICodexCredentialStore

            codex_credential_source = OpenAICodexCredentialStore()
        codex_credential_source.preflight()
        return ResolvedModel(
            model_name=full_model_str,
            model=build_model(
                full_model_str,
                None,
                codex_credential_source=codex_credential_source,
            ),
            provider=provider,
            api_key=None,
        )

    api_key = api_key_override or None
    if api_key is None:
        auth.validate(full_model_str)
        api_key = auth.get_api_key(full_model_str)
    return ResolvedModel(
        model_name=full_model_str,
        model=build_model(full_model_str, api_key),
        provider=provider,
        api_key=api_key,
    )


def resolve_nested_model(
    choice: NestedModel,
    *,
    main: ResolvedModel,
    auth: ModelAuth,
) -> ResolvedModel:
    """Turn a configured nested-model choice into a usable child model."""
    from sqlsaber.nested_model import Pinned

    if isinstance(choice, Pinned):
        return resolve_model(
            auth,
            str(choice.id),
            api_key_override=choice.api_key,
        )
    return resolve_model(
        auth,
        main.model_name,
        api_key_override=main.api_key,
    )
