from __future__ import annotations

import pytest

from sqlsaber import SQLSaber


@pytest.mark.asyncio
async def test_api_system_prompt_text_override(temp_dir, monkeypatch):
    config_dir = temp_dir / "config"
    monkeypatch.setattr(
        "platformdirs.user_config_dir", lambda *args, **kwargs: str(config_dir)
    )

    custom_prompt = "CUSTOM SYSTEM PROMPT"

    saber = SQLSaber(
        database="sqlite:///:memory:",
        model_name="anthropic:claude-3-5-sonnet",
        api_key="test-key",
        system_prompt=custom_prompt,
    )

    try:
        prompt = saber.agent.system_prompt_text(include_memory=True)
        assert custom_prompt in prompt
        assert "You are a helpful SQL assistant" not in prompt
    finally:
        await saber.close()


@pytest.mark.asyncio
async def test_api_system_prompt_file_override(temp_dir, monkeypatch):
    config_dir = temp_dir / "config"
    monkeypatch.setattr(
        "platformdirs.user_config_dir", lambda *args, **kwargs: str(config_dir)
    )

    prompt_file = temp_dir / "system_prompt.txt"
    prompt_file.write_text("prompt from file", encoding="utf-8")

    saber = SQLSaber(
        database="sqlite:///:memory:",
        model_name="anthropic:claude-3-5-sonnet",
        api_key="test-key",
        system_prompt=prompt_file,
    )

    try:
        prompt = saber.agent.system_prompt_text(include_memory=True)
        assert "prompt from file" in prompt
        assert "You are a helpful SQL assistant" not in prompt
    finally:
        await saber.close()


@pytest.mark.asyncio
async def test_api_system_prompt_whitespace_falls_back_to_default(
    temp_dir, monkeypatch
):
    config_dir = temp_dir / "config"
    monkeypatch.setattr(
        "platformdirs.user_config_dir", lambda *args, **kwargs: str(config_dir)
    )

    saber = SQLSaber(
        database="sqlite:///:memory:",
        model_name="anthropic:claude-3-5-sonnet",
        api_key="test-key",
        system_prompt="   \n\t",
    )

    try:
        prompt = saber.agent.system_prompt_text(include_memory=True)
        assert "You are a helpful SQL assistant" in prompt
    finally:
        await saber.close()
