"""Minimal pydantic-ai model construction helpers."""

from typing import Literal

from pydantic_ai.models import Model
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.openai import OpenAIProvider

from sqlsaber.config import providers
from sqlsaber.config.settings import ThinkingLevel

type UnifiedEffort = Literal["minimal", "low", "medium", "high", "xhigh"]

UNIFIED_EFFORT_MAP: dict[ThinkingLevel, UnifiedEffort] = {
    ThinkingLevel.MINIMAL: "minimal",
    ThinkingLevel.LOW: "low",
    ThinkingLevel.MEDIUM: "medium",
    ThinkingLevel.HIGH: "high",
    ThinkingLevel.MAXIMUM: "xhigh",
}


def build_model(full_model_str: str, api_key: str | None) -> Model | str:
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
    return normalized_model_str
