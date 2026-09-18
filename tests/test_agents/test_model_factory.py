"""Tests for model construction and unified thinking settings."""

from typing import Any

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.models.openai_codex import OpenAICodexModel
from pydantic_ai.providers.openai_codex import OpenAICodexCredentials
from pydantic_ai.models.xai import XaiModel

from sqlsaber.agents.model_factory import (
    UNIFIED_EFFORT_MAP,
    build_model,
    resolve_model,
)
from sqlsaber.agents.pydantic_ai_agent import SQLSaberAgent
from sqlsaber.config.settings import ThinkingLevel
from sqlsaber.database.sqlite import SQLiteConnection


@pytest.mark.parametrize(
    ("model_name", "expected_type", "expected_name"),
    [
        ("anthropic:claude-test", AnthropicModel, "claude-test"),
        ("google:gemini-test", GoogleModel, "gemini-test"),
        ("google-gla:gemini-test", GoogleModel, "gemini-test"),
        ("openai:gpt-test", OpenAIResponsesModel, "gpt-test"),
        ("xai:grok-4.6", XaiModel, "grok-4.6"),
    ],
)
def test_build_model_with_explicit_key(
    model_name: str, expected_type: type, expected_name: str
) -> None:
    model = build_model(model_name, "test-key")

    assert isinstance(model, expected_type)
    assert model.model_name == expected_name


def test_build_model_without_key_returns_provider_string() -> None:
    assert build_model("anthropic:claude-test", None) == "anthropic:claude-test"
    assert build_model("xai:grok-4.6", None) == "xai:grok-4.6"


def test_build_model_groq_with_key_stays_discovery_string() -> None:
    assert build_model("groq:llama-3-3-70b-versatile", "test-key") == (
        "groq:llama-3-3-70b-versatile"
    )


def test_build_model_xai_explicit_key_injects_provider() -> None:
    model = build_model("xai:grok-4.6", "test-key")

    assert isinstance(model, XaiModel)
    assert model.model_name == "grok-4.6"
    assert model.system == "xai"


def test_build_model_normalizes_google_alias() -> None:
    assert build_model("google-gla:gemini-test", None) == "google:gemini-test"


class FakeCodexCredentialSource:
    def preflight(self) -> None:
        pass

    async def load(self) -> OpenAICodexCredentials:
        return OpenAICodexCredentials(
            access_token="fake-access",
            refresh_token="fake-refresh",
            account_id="fake-account",
        )

    async def save(self, credentials: OpenAICodexCredentials) -> None:
        del credentials


class NoApiKeyAuth:
    def get_api_key(self, model_name: str) -> str | None:
        raise AssertionError(f"API key requested for {model_name}")

    def validate(self, model_name: str) -> None:
        raise AssertionError(f"API key validation requested for {model_name}")


def test_resolve_openai_codex_uses_explicit_subscription_source_without_key_prompt():
    resolved = resolve_model(
        NoApiKeyAuth(),
        "openai-codex:gpt-test",
        codex_credential_source=FakeCodexCredentialSource(),
    )

    assert resolved.model_name == "openai-codex:gpt-test"
    assert resolved.provider == "openai-codex"
    assert resolved.api_key is None
    assert isinstance(resolved.model, OpenAICodexModel)
    assert resolved.model.model_name == "gpt-test"
    assert resolved.model.system == "openai-codex"


def test_resolve_openai_codex_surfaces_missing_credentials_before_model_request():
    class MissingCredentialSource(FakeCodexCredentialSource):
        def preflight(self) -> None:
            raise ValueError(
                "No SQLsaber OpenAI Codex credentials were found. Run "
                "`saber auth setup openai-codex`."
            )

        async def load(self) -> OpenAICodexCredentials:
            raise AssertionError(
                "HTTP authentication must not load missing credentials"
            )

    with pytest.raises(ValueError, match="saber auth setup openai-codex"):
        resolve_model(
            NoApiKeyAuth(),
            "openai-codex:gpt-test",
            codex_credential_source=MissingCredentialSource(),
        )


def test_build_openai_codex_refuses_implicit_codex_cli_credentials():
    with pytest.raises(ValueError, match="credential source is required"):
        build_model("openai-codex:gpt-test", None)


def test_resolve_openai_api_path_preserves_explicit_key():
    resolved = resolve_model(
        NoApiKeyAuth(),
        "openai:gpt-test",
        api_key_override="test-key",
    )

    assert resolved.provider == "openai"
    assert resolved.api_key == "test-key"
    assert isinstance(resolved.model, OpenAIResponsesModel)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("level", "expected"),
    list(UNIFIED_EFFORT_MAP.items()),
)
async def test_thinking_capability_merges_unified_setting(
    level: ThinkingLevel,
    expected: str,
) -> None:
    captured: dict[str, Any] = {}

    def respond(_messages, info: AgentInfo) -> ModelResponse:
        captured.update(info.model_settings or {})
        return ModelResponse(parts=[TextPart(content="ok")])

    wrapper = SQLSaberAgent(
        db_connection=SQLiteConnection("sqlite:///:memory:"),
        model_name="anthropic:claude-test",
        api_key="test-key",
        thinking_enabled=True,
        thinking_level=level,
    )
    with wrapper.agent.override(model=FunctionModel(respond)):
        await wrapper.run("Hello")

    # FunctionModel strips the provider-agnostic setting before invoking its
    # callback, so inspect the capability settings before model preparation.
    assert wrapper.agent._cap_model_settings["thinking"] == expected
    assert captured["anthropic_cache"] is True
    await wrapper.close()


@pytest.mark.asyncio
async def test_thinking_disabled_omits_unified_setting() -> None:
    captured: dict[str, Any] = {}

    def respond(_messages, info: AgentInfo) -> ModelResponse:
        captured.update(info.model_settings or {})
        return ModelResponse(parts=[TextPart(content="ok")])

    wrapper = SQLSaberAgent(
        db_connection=SQLiteConnection("sqlite:///:memory:"),
        model_name="openai:gpt-test",
        api_key="test-key",
        thinking_enabled=False,
    )
    with wrapper.agent.override(model=FunctionModel(respond)):
        await wrapper.run("Hello")

    assert "thinking" not in captured
    await wrapper.close()
