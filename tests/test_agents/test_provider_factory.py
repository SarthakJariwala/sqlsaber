import pytest
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIResponsesModel

from sqlsaber.agents.provider_factory import (
    AnthropicOAuthProviderStrategy,
    AnthropicProviderStrategy,
    DefaultProviderStrategy,
    GoogleProviderStrategy,
    GroqProviderStrategy,
    OpenAIProviderStrategy,
    ProviderFactory,
)


@pytest.fixture
def factory():
    return ProviderFactory()


def test_strategies_map(factory):
    """Test that the factory returns the correct strategy for each provider."""
    assert isinstance(factory.get_strategy("google"), GoogleProviderStrategy)
    assert isinstance(factory.get_strategy("openai"), OpenAIProviderStrategy)
    assert isinstance(factory.get_strategy("groq"), GroqProviderStrategy)
    assert isinstance(
        factory.get_strategy("anthropic", is_oauth=True), AnthropicOAuthProviderStrategy
    )
    assert isinstance(
        factory.get_strategy("anthropic", is_oauth=False), AnthropicProviderStrategy
    )
    assert isinstance(factory.get_strategy("unknown"), DefaultProviderStrategy)


def test_google_strategy_real():
    """Test creating a Google agent with real objects."""
    strategy = GoogleProviderStrategy()
    agent = strategy.create_agent(
        model_name="gemini-pro", api_key="dummy-key", thinking_enabled=True
    )

    assert isinstance(agent, Agent)
    assert isinstance(agent.model, GoogleModel)
    assert agent.model.model_name == "gemini-pro"

    # Verify thinking settings
    settings = agent.model_settings
    # GoogleModelSettings is a TypedDict, so we check it as a dict
    assert settings
    assert settings.get("google_thinking_config", {}).get("include_thoughts") is True


def test_anthropic_oauth_strategy_real():
    """Test creating an Anthropic OAuth agent with real objects."""
    strategy = AnthropicOAuthProviderStrategy()

    # Verify token requirement
    with pytest.raises(ValueError):
        strategy.create_agent("claude-3")

    agent = strategy.create_agent(
        model_name="claude-3", oauth_token="oauth-token", thinking_enabled=True
    )

    assert isinstance(agent, Agent)
    assert isinstance(agent.model, AnthropicModel)
    assert agent.model.model_name == "claude-3"

    # Verify thinking settings
    settings = agent.model_settings
    # AnthropicModelSettings is a TypedDict
    assert settings
    assert settings.get("anthropic_thinking", {}).get("type") == "enabled"
    assert settings.get("anthropic_thinking", {}).get("budget_tokens") == 2048


def test_anthropic_strategy_real(monkeypatch):
    """Test creating a standard Anthropic agent."""
    strategy = AnthropicProviderStrategy()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")

    agent = strategy.create_agent(
        model_name="anthropic:claude-3", thinking_enabled=True
    )

    assert isinstance(agent, Agent)
    # When using model_name string, checking specific model type is harder
    # without inspection, but we verify agent creation success.

    settings = agent.model_settings
    # AnthropicModelSettings is a TypedDict
    assert settings
    assert settings.get("anthropic_thinking", {}).get("type") == "enabled"


def test_openai_strategy_real(monkeypatch):
    """Test creating an OpenAI agent with real objects."""
    strategy = OpenAIProviderStrategy()
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")

    agent = strategy.create_agent(model_name="gpt-4", thinking_enabled=True)

    assert isinstance(agent, Agent)
    assert isinstance(agent.model, OpenAIResponsesModel)
    assert agent.model.model_name == "gpt-4"

    settings = agent.model_settings
    # OpenAIResponsesModelSettings is a TypedDict
    assert settings
    assert settings.get("openai_reasoning_effort") == "medium"


def test_groq_strategy_real(monkeypatch):
    """Test creating a Groq agent with real objects."""
    strategy = GroqProviderStrategy()
    monkeypatch.setenv("GROQ_API_KEY", "dummy")

    agent = strategy.create_agent(model_name="groq:llama-3", thinking_enabled=True)

    assert isinstance(agent, Agent)

    settings = agent.model_settings
    # GroqModelSettings is a TypedDict
    assert settings
    assert settings.get("groq_reasoning_format") == "parsed"


def test_factory_create_agent_integration(factory, monkeypatch):
    """Test the factory's create_agent method end-to-end."""
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy")

    # Test Google path
    agent = factory.create_agent(
        provider="google",
        model_name="gemini-pro",
        full_model_str="google:gemini-pro",
        api_key="dummy-key",
    )
    assert isinstance(agent.model, GoogleModel)
    assert agent.model.model_name == "gemini-pro"


# Tests for explicit API key usage (no env var required)


def test_anthropic_strategy_with_explicit_api_key():
    """Test Anthropic strategy creates agent with explicit api_key (no env var)."""
    strategy = AnthropicProviderStrategy()
    agent = strategy.create_agent(model_name="claude-3", api_key="test-api-key")

    assert isinstance(agent, Agent)
    assert isinstance(agent.model, AnthropicModel)
    assert agent.model.model_name == "claude-3"


def test_openai_strategy_with_explicit_api_key():
    """Test OpenAI strategy creates agent with explicit api_key (no env var)."""
    strategy = OpenAIProviderStrategy()
    agent = strategy.create_agent(model_name="gpt-4", api_key="test-api-key")

    assert isinstance(agent, Agent)
    assert isinstance(agent.model, OpenAIResponsesModel)
    assert agent.model.model_name == "gpt-4"


def test_google_strategy_with_explicit_api_key():
    """Test Google strategy creates agent with explicit api_key (no env var)."""
    strategy = GoogleProviderStrategy()
    agent = strategy.create_agent(model_name="gemini-pro", api_key="test-api-key")

    assert isinstance(agent, Agent)
    assert isinstance(agent.model, GoogleModel)
    assert agent.model.model_name == "gemini-pro"
