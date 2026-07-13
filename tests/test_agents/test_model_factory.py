"""Tests for model construction and unified thinking settings."""

from typing import Any

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIResponsesModel

from sqlsaber.agents.model_factory import UNIFIED_EFFORT_MAP, build_model
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


def test_build_model_normalizes_google_alias() -> None:
    assert build_model("google-gla:gemini-test", None) == "google:gemini-test"


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
