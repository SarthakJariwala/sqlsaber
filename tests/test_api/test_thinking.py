from __future__ import annotations

import pytest

from sqlsaber import SQLSaber


@pytest.mark.asyncio
async def test_api_thinking_enabled(temp_dir, monkeypatch):
    """Test that thinking=True enables thinking mode on the agent."""
    config_dir = temp_dir / "config"
    monkeypatch.setattr(
        "platformdirs.user_config_dir", lambda *args, **kwargs: str(config_dir)
    )

    saber = SQLSaber(
        database="sqlite:///:memory:",
        model_name="anthropic:claude-3-5-sonnet",
        api_key="test-key",
        thinking=True,
    )

    try:
        assert saber.agent.thinking_enabled is True
    finally:
        await saber.close()


@pytest.mark.asyncio
async def test_api_thinking_disabled_by_default(temp_dir, monkeypatch):
    """Test that thinking is disabled by default."""
    config_dir = temp_dir / "config"
    monkeypatch.setattr(
        "platformdirs.user_config_dir", lambda *args, **kwargs: str(config_dir)
    )

    saber = SQLSaber(
        database="sqlite:///:memory:",
        model_name="anthropic:claude-3-5-sonnet",
        api_key="test-key",
    )

    try:
        assert saber.agent.thinking_enabled is False
    finally:
        await saber.close()


@pytest.mark.asyncio
async def test_api_thinking_explicit_false(temp_dir, monkeypatch):
    """Test that thinking=False explicitly disables thinking mode."""
    config_dir = temp_dir / "config"
    monkeypatch.setattr(
        "platformdirs.user_config_dir", lambda *args, **kwargs: str(config_dir)
    )

    saber = SQLSaber(
        database="sqlite:///:memory:",
        model_name="anthropic:claude-3-5-sonnet",
        api_key="test-key",
        thinking=False,
    )

    try:
        assert saber.agent.thinking_enabled is False
    finally:
        await saber.close()
