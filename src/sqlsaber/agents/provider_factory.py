"""Provider Factory for creating Pydantic-AI Agents.

This module implements the Factory and Strategy patterns to handle the creation of
agents for different providers (Google, Anthropic, OpenAI, etc.), encapsulating
provider-specific logic and configuration.
"""

import abc
from typing import Any, Literal, cast, override

import httpx
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.models.groq import GroqModelSettings
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.openai import OpenAIProvider

ProviderName = Literal["google", "anthropic", "anthropic_oauth", "openai", "groq"]


class AgentProviderStrategy(abc.ABC):
    """Abstract base class for provider-specific agent creation strategies."""

    @abc.abstractmethod
    def create_agent(
        self,
        model_name: str,
        api_key: str | None = None,
        oauth_token: str | None = None,
        thinking_enabled: bool = False,
    ) -> Agent:
        """Create and configure an Agent for this provider."""
        pass


class GoogleProviderStrategy(AgentProviderStrategy):
    """Strategy for creating Google agents."""

    @override
    def create_agent(
        self,
        model_name: str,
        api_key: str | None = None,
        oauth_token: str | None = None,
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


class AnthropicOAuthProviderStrategy(AgentProviderStrategy):
    """Strategy for creating Anthropic agents with OAuth."""

    @override
    def create_agent(
        self,
        model_name: str,
        api_key: str | None = None,
        oauth_token: str | None = None,
        thinking_enabled: bool = False,
    ) -> Agent:
        if not oauth_token:
            raise ValueError("OAuth token is required for Anthropic OAuth strategy.")

        async def add_oauth_headers(request: httpx.Request) -> None:
            if "x-api-key" in request.headers:
                del request.headers["x-api-key"]
            request.headers.update(
                {
                    "Authorization": f"Bearer {oauth_token}",
                    "anthropic-version": "2023-06-01",
                    "anthropic-beta": "oauth-2025-04-20",
                    "User-Agent": "ClaudeCode/1.0 (Anthropic Claude Code CLI)",
                    "X-Client-Name": "claude-code",
                    "X-Client-Version": "1.0.0",
                }
            )

        http_client = httpx.AsyncClient(event_hooks={"request": [add_oauth_headers]})
        provider_obj = AnthropicProvider(api_key="placeholder", http_client=http_client)
        model_obj = AnthropicModel(model_name, provider=provider_obj)

        if thinking_enabled:
            settings = AnthropicModelSettings(
                anthropic_thinking=cast(
                    Any, {"type": "enabled", "budget_tokens": 2048}
                ),
                max_tokens=8192,
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
        oauth_token: str | None = None,
        thinking_enabled: bool = False,
    ) -> Agent:
        # Use explicit provider if api_key provided, else let pydantic-ai use env var
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
        oauth_token: str | None = None,
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
        oauth_token: str | None = None,
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
        oauth_token: str | None = None,
        thinking_enabled: bool = False,
    ) -> Agent:
        return Agent(model_name, name="sqlsaber")


class ProviderFactory:
    """Factory to create agents based on provider configuration."""

    def __init__(self) -> None:
        self._strategies: dict[str, AgentProviderStrategy] = {
            "google": GoogleProviderStrategy(),
            "anthropic_oauth": AnthropicOAuthProviderStrategy(),
            "anthropic": AnthropicProviderStrategy(),
            "openai": OpenAIProviderStrategy(),
            "groq": GroqProviderStrategy(),
        }
        self._default_strategy: AgentProviderStrategy = DefaultProviderStrategy()

    def get_strategy(
        self, provider: ProviderName | str, is_oauth: bool = False
    ) -> AgentProviderStrategy:
        """Retrieve the appropriate strategy for the provider."""
        if provider == "anthropic" and is_oauth:
            return self._strategies["anthropic_oauth"]

        return self._strategies.get(provider, self._default_strategy)

    def create_agent(
        self,
        provider: ProviderName | str,
        model_name: str,
        full_model_str: str,
        api_key: str | None = None,
        oauth_token: str | None = None,
        thinking_enabled: bool = False,
        is_oauth: bool = False,
    ) -> Agent:
        """Create an agent using the appropriate strategy.

        Args:
            provider: The provider key (e.g., 'google', 'anthropic').
            model_name: The model name stripped of provider prefix (e.g., 'gemini-1.5-pro').
            full_model_str: The full model configuration string (e.g., 'anthropic:claude-3-5-sonnet').
            api_key: Optional API key.
            oauth_token: Optional OAuth token.
            thinking_enabled: Whether to enable thinking/reasoning features.
            is_oauth: Whether to use OAuth (specifically for Anthropic).
        """
        strategy = self.get_strategy(provider, is_oauth)

        target_name = full_model_str

        if isinstance(
            strategy,
            (
                GoogleProviderStrategy,
                OpenAIProviderStrategy,
                AnthropicOAuthProviderStrategy,
                AnthropicProviderStrategy,
            ),
        ):
            target_name = model_name

        return strategy.create_agent(
            model_name=target_name,
            api_key=api_key,
            oauth_token=oauth_token,
            thinking_enabled=thinking_enabled,
        )
