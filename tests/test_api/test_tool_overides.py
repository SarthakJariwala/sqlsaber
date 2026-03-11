from __future__ import annotations

import pytest

from sqlsaber import ModelOverides, SQLSaber, SQLSaberOptions


@pytest.mark.asyncio
async def test_api_tool_overides_are_normalized(temp_dir, monkeypatch):
    config_dir = temp_dir / "config"
    monkeypatch.setattr(
        "platformdirs.user_config_dir", lambda *args, **kwargs: str(config_dir)
    )

    saber = SQLSaber(
        options=SQLSaberOptions(
            database="sqlite:///:memory:",
            model_name="anthropic:claude-3-5-sonnet",
            api_key="test-key",
            tool_overrides={
                " viz ": ModelOverides(
                    model_name=" openai:gpt-5-mini ",
                    api_key=" sk-test ",
                )
            },
        )
    )

    try:
        assert saber.agent._tool_overides["viz"].model_name == "openai:gpt-5-mini"
        assert saber.agent._tool_overides["viz"].api_key == "sk-test"
    finally:
        await saber.close()


def test_api_tool_overides_reject_invalid_api_key_without_model(temp_dir, monkeypatch):
    config_dir = temp_dir / "config"
    monkeypatch.setattr(
        "platformdirs.user_config_dir", lambda *args, **kwargs: str(config_dir)
    )

    with pytest.raises(ValueError, match="api_key override requires model_name"):
        SQLSaber(
            options=SQLSaberOptions(
                database="sqlite:///:memory:",
                model_name="anthropic:claude-3-5-sonnet",
                api_key="test-key",
                tool_overrides={"viz": {"api_key": "sk-test"}},
            )
        )


def test_api_tool_overides_legacy_keyword_is_rejected() -> None:
    with pytest.raises(TypeError, match="tool_overides"):
        SQLSaber(
            options=SQLSaberOptions(database="sqlite:///:memory:"),
            **{
                "tool_overides": {"viz": {"model_name": "openai:gpt-5-mini"}},
            },
        )
