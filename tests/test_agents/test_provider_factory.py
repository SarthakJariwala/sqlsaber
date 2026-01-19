import pytest
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIResponsesModel

from sqlsaber.agents.provider_factory import (
    ANTHROPIC_BUDGET_MAP,
    GOOGLE_LEVEL_MAP,
    OPENAI_EFFORT_MAP,
    AnthropicProviderStrategy,
    DefaultProviderStrategy,
    GoogleProviderStrategy,
    GroqProviderStrategy,
    OpenAIProviderStrategy,
    ProviderFactory,
)
from sqlsaber.config.settings import ThinkingLevel


@pytest.fixture
def factory():
    return ProviderFactory()


def test_strategies_map(factory):
    """Test that the factory returns the correct strategy for each provider."""
    assert isinstance(factory.get_strategy("google"), GoogleProviderStrategy)
    assert isinstance(factory.get_strategy("openai"), OpenAIProviderStrategy)
    assert isinstance(factory.get_strategy("groq"), GroqProviderStrategy)
    assert isinstance(factory.get_strategy("anthropic"), AnthropicProviderStrategy)
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

    settings = agent.model_settings
    assert settings
    assert settings.get("google_thinking_config", {}).get("include_thoughts") is True


def test_anthropic_strategy_real(monkeypatch):
    """Test creating a standard Anthropic agent."""
    strategy = AnthropicProviderStrategy()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")

    agent = strategy.create_agent(
        model_name="anthropic:claude-3", thinking_enabled=True
    )

    assert isinstance(agent, Agent)

    settings = agent.model_settings
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
    assert settings
    assert settings.get("openai_reasoning_effort") == "medium"


def test_groq_strategy_real(monkeypatch):
    """Test creating a Groq agent with real objects."""
    strategy = GroqProviderStrategy()
    monkeypatch.setenv("GROQ_API_KEY", "dummy")

    agent = strategy.create_agent(model_name="groq:llama-3", thinking_enabled=True)

    assert isinstance(agent, Agent)

    settings = agent.model_settings
    assert settings
    assert settings.get("groq_reasoning_format") == "parsed"


def test_factory_create_agent_integration(factory, monkeypatch):
    """Test the factory's create_agent method end-to-end."""
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy")

    agent = factory.create_agent(
        provider="google",
        model_name="gemini-pro",
        full_model_str="google:gemini-pro",
        api_key="dummy-key",
    )
    assert isinstance(agent.model, GoogleModel)
    assert agent.model.model_name == "gemini-pro"


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


class TestAnthropicThinkingLevels:
    """Test Anthropic strategy with different thinking levels."""

    @pytest.mark.parametrize(
        "level,expected_budget",
        [
            (ThinkingLevel.MINIMAL, 1024),
            (ThinkingLevel.LOW, 2048),
            (ThinkingLevel.MEDIUM, 8192),
            (ThinkingLevel.HIGH, 32768),
            (ThinkingLevel.MAXIMUM, 100000),
        ],
    )
    def test_anthropic_thinking_level_mapping(
        self, monkeypatch, level, expected_budget
    ):
        """Test Anthropic budget_tokens mapping for each level."""
        strategy = AnthropicProviderStrategy()
        monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")

        agent = strategy.create_agent(
            model_name="claude-sonnet-4",
            thinking_enabled=True,
            thinking_level=level,
        )

        settings = agent.model_settings
        assert settings
        assert (
            settings.get("anthropic_thinking", {}).get("budget_tokens")
            == expected_budget
        )

    def test_anthropic_thinking_disabled(self, monkeypatch):
        """Test that disabling thinking skips configuration."""
        strategy = AnthropicProviderStrategy()
        monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")

        agent = strategy.create_agent(
            model_name="claude-sonnet-4",
            thinking_enabled=False,
            thinking_level=ThinkingLevel.HIGH,
        )

        settings = agent.model_settings
        assert not settings or "anthropic_thinking" not in settings

    def test_budget_map_completeness(self):
        """Test that all levels have budget mappings."""
        for level in ThinkingLevel:
            assert level in ANTHROPIC_BUDGET_MAP, f"Missing budget mapping for {level}"


class TestOpenAIThinkingLevels:
    """Test OpenAI strategy with different thinking levels."""

    @pytest.mark.parametrize(
        "level,expected_effort",
        [
            (ThinkingLevel.MINIMAL, "minimal"),
            (ThinkingLevel.LOW, "low"),
            (ThinkingLevel.MEDIUM, "medium"),
            (ThinkingLevel.HIGH, "high"),
            (ThinkingLevel.MAXIMUM, "xhigh"),
        ],
    )
    def test_openai_thinking_level_mapping(self, monkeypatch, level, expected_effort):
        """Test OpenAI reasoning_effort mapping for each level."""
        strategy = OpenAIProviderStrategy()
        monkeypatch.setenv("OPENAI_API_KEY", "dummy")

        agent = strategy.create_agent(
            model_name="gpt-4",
            thinking_enabled=True,
            thinking_level=level,
        )

        settings = agent.model_settings
        assert settings
        assert settings.get("openai_reasoning_effort") == expected_effort

    def test_openai_thinking_disabled(self, monkeypatch):
        """Test that disabling thinking skips configuration."""
        strategy = OpenAIProviderStrategy()
        monkeypatch.setenv("OPENAI_API_KEY", "dummy")

        agent = strategy.create_agent(
            model_name="gpt-4",
            thinking_enabled=False,
            thinking_level=ThinkingLevel.HIGH,
        )

        settings = agent.model_settings
        assert not settings or "openai_reasoning_effort" not in settings

    def test_effort_map_completeness(self):
        """Test that all levels have effort mappings."""
        for level in ThinkingLevel:
            assert level in OPENAI_EFFORT_MAP, f"Missing effort mapping for {level}"


class TestGoogleThinkingLevels:
    """Test Google strategy with different thinking levels."""

    @pytest.mark.parametrize(
        "level,expected_google_level",
        [
            (ThinkingLevel.MINIMAL, "MINIMAL"),
            (ThinkingLevel.LOW, "LOW"),
            (ThinkingLevel.MEDIUM, "MEDIUM"),
            (ThinkingLevel.HIGH, "HIGH"),
            (ThinkingLevel.MAXIMUM, "HIGH"),  # Google caps at HIGH
        ],
    )
    def test_google_thinking_level_mapping(self, level, expected_google_level):
        """Test Google thinking_level mapping for each level."""
        strategy = GoogleProviderStrategy()

        agent = strategy.create_agent(
            model_name="gemini-pro",
            api_key="dummy-key",
            thinking_enabled=True,
            thinking_level=level,
        )

        settings = agent.model_settings
        assert settings
        thinking_config = settings.get("google_thinking_config", {})
        assert thinking_config.get("include_thoughts") is True
        assert thinking_config.get("thinking_level") == expected_google_level

    def test_google_thinking_disabled(self):
        """Test that disabling thinking skips configuration."""
        strategy = GoogleProviderStrategy()

        agent = strategy.create_agent(
            model_name="gemini-pro",
            api_key="dummy-key",
            thinking_enabled=False,
            thinking_level=ThinkingLevel.HIGH,
        )

        settings = agent.model_settings
        assert not settings or "google_thinking_config" not in settings

    def test_level_map_completeness(self):
        """Test that all levels have Google level mappings."""
        for level in ThinkingLevel:
            assert level in GOOGLE_LEVEL_MAP, (
                f"Missing Google level mapping for {level}"
            )


class TestGroqThinkingLevels:
    """Test Groq strategy with thinking levels (binary only)."""

    @pytest.mark.parametrize(
        "level",
        [
            ThinkingLevel.MINIMAL,
            ThinkingLevel.LOW,
            ThinkingLevel.MEDIUM,
            ThinkingLevel.HIGH,
            ThinkingLevel.MAXIMUM,
        ],
    )
    def test_groq_any_level_enables_thinking(self, monkeypatch, level):
        """Test that any level enables Groq reasoning when thinking is on."""
        strategy = GroqProviderStrategy()
        monkeypatch.setenv("GROQ_API_KEY", "dummy")

        agent = strategy.create_agent(
            model_name="groq:llama-3",
            thinking_enabled=True,
            thinking_level=level,
        )

        settings = agent.model_settings
        assert settings
        assert settings.get("groq_reasoning_format") == "parsed"

    def test_groq_thinking_disabled(self, monkeypatch):
        """Test that disabling thinking skips configuration."""
        strategy = GroqProviderStrategy()
        monkeypatch.setenv("GROQ_API_KEY", "dummy")

        agent = strategy.create_agent(
            model_name="groq:llama-3",
            thinking_enabled=False,
            thinking_level=ThinkingLevel.HIGH,
        )

        settings = agent.model_settings
        assert not settings or "groq_reasoning_format" not in settings


class TestProviderFactoryWithLevels:
    """Test ProviderFactory.create_agent with thinking levels."""

    def test_factory_passes_thinking_level(self, monkeypatch):
        """Test that factory passes thinking_level to strategies."""
        factory = ProviderFactory()
        monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")

        agent = factory.create_agent(
            provider="anthropic",
            model_name="claude-sonnet-4",
            full_model_str="anthropic:claude-sonnet-4",
            thinking_enabled=True,
            thinking_level=ThinkingLevel.HIGH,
        )

        settings = agent.model_settings
        assert settings
        assert settings.get("anthropic_thinking", {}).get("budget_tokens") == 32768
