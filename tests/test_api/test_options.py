from __future__ import annotations

import pytest

from sqlsaber import SQLSaber, SQLSaberOptions
from sqlsaber.knowledge.manager import KnowledgeManager
from sqlsaber.knowledge.sqlite_store import SQLiteKnowledgeStore


@pytest.mark.asyncio
async def test_api_options_preferred_over_legacy_args(temp_dir, monkeypatch):
    config_dir = temp_dir / "config"
    monkeypatch.setattr(
        "platformdirs.user_config_dir", lambda *args, **kwargs: str(config_dir)
    )

    options = SQLSaberOptions(
        database="sqlite:///:memory:",
        model_name="anthropic:claude-3-5-sonnet",
        api_key="options-key",
        thinking_enabled=False,
    )

    saber = SQLSaber(
        database="definitely-not-a-valid-config-name",
        model_name="openai:gpt-5-mini",
        api_key="legacy-key",
        thinking=True,
        options=options,
    )

    try:
        assert saber.agent.agent.model.model_name == "claude-3-5-sonnet"
        assert saber.agent.thinking_enabled is False
    finally:
        await saber.close()


@pytest.mark.asyncio
async def test_api_close_does_not_close_injected_knowledge_manager(
    temp_dir, monkeypatch
):
    config_dir = temp_dir / "config"
    monkeypatch.setattr(
        "platformdirs.user_config_dir", lambda *args, **kwargs: str(config_dir)
    )

    manager = KnowledgeManager(
        store=SQLiteKnowledgeStore(db_path=temp_dir / "knowledge.db")
    )
    options = SQLSaberOptions(
        database="sqlite:///:memory:",
        model_name="anthropic:claude-3-5-sonnet",
        api_key="test-key",
        knowledge_manager=manager,
    )
    saber = SQLSaber(options=options)

    close_calls = 0

    async def _track_close() -> None:
        nonlocal close_calls
        close_calls += 1

    monkeypatch.setattr(manager, "close", _track_close)

    await saber.close()
    await saber.close()

    assert close_calls == 0


@pytest.mark.asyncio
async def test_api_close_closes_owned_knowledge_manager_once(temp_dir, monkeypatch):
    config_dir = temp_dir / "config"
    monkeypatch.setattr(
        "platformdirs.user_config_dir", lambda *args, **kwargs: str(config_dir)
    )

    options = SQLSaberOptions(
        database="sqlite:///:memory:",
        model_name="anthropic:claude-3-5-sonnet",
        api_key="test-key",
    )
    saber = SQLSaber(options=options)

    close_calls = 0

    async def _track_close() -> None:
        nonlocal close_calls
        close_calls += 1

    monkeypatch.setattr(saber.agent.knowledge_manager, "close", _track_close)

    await saber.close()
    await saber.close()

    assert close_calls == 1
