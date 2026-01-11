"""Provider Factory for creating Pydantic-AI Agents.

This module implements the Factory and Strategy patterns to handle the creation of
agents for different providers (Google, Anthropic, OpenAI, etc.), encapsulating
provider-specific logic and configuration.
"""

import abc
from typing import Any, Literal, cast, override

from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.models.groq import GroqModelSettings
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.openai import OpenAIProvider

ProviderName = Literal["google", "anthropic", "openai", "groq"]


class AgentProviderStrategy(abc.ABC):
    """Abstract base class for provider-specific agent creation strategies."""

    @abc.abstractmethod
    def create_agent(
        self,
        model_name: str,
        api_key: str | None = None,
        thinking_enabled: bool = False,
    ) -> Agent:
        """Create and configure an Agent for this provider."""


class GoogleProviderStrategy(AgentProviderStrategy):
    """Strategy for creating Google agents."""

    @override
    def create_agent(
        self,
        model_name: str,
        api_key: str | None = None,
        thinking_enabled: bool = False,
    ) -> Agent:
        if api_key:
            model_obj = GoogleModel(
                model_name, provider=GoogleProvider(api_key=api_key)
            )
        else:
            model_obj = GoogleModel(model_name)

        if thinking_enabled:
            settings = GoogleModelSettings(
                google_thinking_config={"include_thoughts": True}
            )
            return Agent(model_obj, name="sqlsaber", model_settings=settings)

        return Agent(model_obj, name="sqlsaber")


class AnthropicProviderStrategy(AgentProviderStrategy):
    """Strategy for creating standard Anthropic agents."""

    @override
    def create_agent(
        self,
        model_name: str,
        api_key: str | None = None,
        thinking_enabled: bool = False,
    ) -> Agent:
        if api_key:
            model_obj = AnthropicModel(
                model_name, provider=AnthropicProvider(api_key=api_key)
            )
        else:
            model_obj = AnthropicModel(model_name)

        if thinking_enabled:
            settings = AnthropicModelSettings(
                anthropic_thinking=cast(
                    Any, {"type": "enabled", "budget_tokens": 2048}
                ),
                max_tokens=8192,
            )
            return Agent(model_obj, name="sqlsaber", model_settings=settings)

        return Agent(model_obj, name="sqlsaber")


class OpenAIProviderStrategy(AgentProviderStrategy):
    """Strategy for creating OpenAI agents."""

    @override
    def create_agent(
        self,
        model_name: str,
        api_key: str | None = None,
        thinking_enabled: bool = False,
    ) -> Agent:
        if api_key:
            model_obj = OpenAIResponsesModel(
                model_name, provider=OpenAIProvider(api_key=api_key)
            )
        else:
            model_obj = OpenAIResponsesModel(model_name)

        if thinking_enabled:
            settings = OpenAIResponsesModelSettings(
                openai_reasoning_effort="medium",
                openai_reasoning_summary=cast(Any, "auto"),
            )
            return Agent(model_obj, name="sqlsaber", model_settings=settings)

        return Agent(model_obj, name="sqlsaber")


class GroqProviderStrategy(AgentProviderStrategy):
    """Strategy for creating Groq agents."""

    @override
    def create_agent(
        self,
        model_name: str,
        api_key: str | None = None,
        thinking_enabled: bool = False,
    ) -> Agent:
        if thinking_enabled:
            settings = GroqModelSettings(groq_reasoning_format="parsed")
            return Agent(model_name, name="sqlsaber", model_settings=settings)

        return Agent(model_name, name="sqlsaber")


class DefaultProviderStrategy(AgentProviderStrategy):
    """Default strategy for other providers."""

    @override
    def create_agent(
        self,
        model_name: str,
        api_key: str | None = None,
        thinking_enabled: bool = False,
    ) -> Agent:
        return Agent(model_name, name="sqlsaber")


class ProviderFactory:
    """Factory to create agents based on provider configuration."""

    def __init__(self) -> None:
        self._strategies: dict[str, AgentProviderStrategy] = {
            "google": GoogleProviderStrategy(),
            "anthropic": AnthropicProviderStrategy(),
            "openai": OpenAIProviderStrategy(),
            "groq": GroqProviderStrategy(),
        }
        self._default_strategy: AgentProviderStrategy = DefaultProviderStrategy()

    def get_strategy(self, provider: ProviderName | str) -> AgentProviderStrategy:
        """Retrieve the appropriate strategy for the provider."""
        return self._strategies.get(provider, self._default_strategy)

    def create_agent(
        self,
        provider: ProviderName | str,
        model_name: str,
        full_model_str: str,
        api_key: str | None = None,
        thinking_enabled: bool = False,
    ) -> Agent:
        """Create an agent using the appropriate strategy.

        Args:
            provider: The provider key (e.g., 'google', 'anthropic').
            model_name: The model name stripped of provider prefix (e.g., 'gemini-1.5-pro').
            full_model_str: The full model configuration string (e.g., 'anthropic:claude-3-5-sonnet').
            api_key: Optional API key.
            thinking_enabled: Whether to enable thinking/reasoning features.
        """
        strategy = self.get_strategy(provider)

        target_name = full_model_str
        if isinstance(
            strategy,
            (GoogleProviderStrategy, OpenAIProviderStrategy, AnthropicProviderStrategy),
        ):
            target_name = model_name

        return strategy.create_agent(
            model_name=target_name,
            api_key=api_key,
            thinking_enabled=thinking_enabled,
        )
