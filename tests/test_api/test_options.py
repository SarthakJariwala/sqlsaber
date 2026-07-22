from __future__ import annotations

import pytest
from pydantic_ai.capabilities import Capability

from sqlsaber import (
    InMemoryArtifactPublisher,
    Knowledge,
    SQLSaber,
    SQLSaberOptions,
    SqlTools,
)
from sqlsaber.config.settings import Config
from sqlsaber.knowledge.manager import KnowledgeManager
from sqlsaber.knowledge.sqlite_store import SQLiteKnowledgeStore


def test_api_options_are_required() -> None:
    kwargs: dict[str, object] = {}
    with pytest.raises(TypeError, match="options"):
        SQLSaber(**kwargs)


def test_capabilities_are_exported_from_top_level() -> None:
    assert SqlTools.__name__ == "SqlTools"
    assert Knowledge.__name__ == "Knowledge"
    assert InMemoryArtifactPublisher.__name__ == "InMemoryArtifactPublisher"


def test_invalid_artifact_failure_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="artifact_failure_mode"):
        SQLSaber(
            options=SQLSaberOptions(
                database="sqlite:///:memory:",
                artifact_failure_mode="invalid",  # type: ignore[arg-type]
            )
        )


@pytest.mark.parametrize(
    ("legacy_kw", "value"),
    [
        ("database", "sqlite:///:memory:"),
        ("thinking", True),
        ("thinking_level", "high"),
        ("model_name", "anthropic:claude-3-5-sonnet"),
        ("api_key", "test-key"),
        ("system_prompt", "custom system prompt"),
        ("tool_overrides", {"viz": {"model_name": "openai:gpt-5-mini"}}),
        ("knowledge_manager", object()),
    ],
)
def test_api_legacy_constructor_kwargs_are_rejected(
    legacy_kw: str, value: object
) -> None:
    kwargs: dict[str, object] = {legacy_kw: value}
    with pytest.raises(TypeError, match=legacy_kw):
        SQLSaber(**kwargs)


@pytest.mark.asyncio
async def test_extra_capabilities_are_appended_to_managed_agent() -> None:
    extra = Capability(id="custom", instructions="Custom capability")
    saber = SQLSaber(
        options=SQLSaberOptions(
            database="sqlite:///:memory:",
            settings=Config.in_memory(
                model_name="anthropic:claude-3-5-sonnet",
                api_keys={"anthropic": "test-key"},
            ),
            extra_capabilities=[extra],
        )
    )

    assert extra in saber.agent.capabilities
    await saber.close()


@pytest.mark.asyncio
async def test_artifact_options_and_run_context_reach_managed_agent(
    monkeypatch,
) -> None:
    publisher = InMemoryArtifactPublisher()
    saber = SQLSaber(
        options=SQLSaberOptions(
            database="sqlite:///:memory:",
            settings=Config.in_memory(
                model_name="anthropic:claude-3-5-sonnet",
                api_keys={"anthropic": "test-key"},
            ),
            artifact_publisher=publisher,
            artifact_failure_mode="best_effort",
        )
    )
    captured: dict[str, object] = {}

    class FakeResult:
        output = "answer"

        def new_messages(self):
            return []

        def all_messages(self):
            return []

    async def fake_run(prompt: str, **kwargs):
        captured.update(prompt=prompt, **kwargs)
        return FakeResult()

    monkeypatch.setattr(saber.agent, "run", fake_run)
    result = await saber.query(
        "analyze",
        conversation_id="conversation-1",
        metadata={"tenant_id": "acme"},
    )

    assert result == "answer"
    assert saber.agent._artifact_publisher is publisher
    assert saber.agent._artifact_failure_mode == "best_effort"
    assert captured["conversation_id"] == "conversation-1"
    assert captured["metadata"] == {"tenant_id": "acme"}
    await saber.close()


@pytest.mark.asyncio
async def test_api_close_does_not_close_injected_knowledge_manager(
    temp_dir, monkeypatch
):
    manager = KnowledgeManager(
        store=SQLiteKnowledgeStore(db_path=temp_dir / "knowledge.db")
    )
    options = SQLSaberOptions(
        database="sqlite:///:memory:",
        settings=Config.in_memory(
            model_name="anthropic:claude-3-5-sonnet",
            api_keys={"anthropic": "test-key"},
        ),
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
async def test_api_close_closes_owned_knowledge_manager_once(monkeypatch):
    options = SQLSaberOptions(
        database="sqlite:///:memory:",
        settings=Config.in_memory(
            model_name="anthropic:claude-3-5-sonnet",
            api_keys={"anthropic": "test-key"},
        ),
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


@pytest.mark.asyncio
async def test_api_in_memory_settings_avoid_filesystem_side_effects(monkeypatch):
    def _fail_model_config_manager(*args, **kwargs):
        _ = args, kwargs
        raise AssertionError("ModelConfigManager should not be constructed")

    def _fail_database_config_manager(*args, **kwargs):
        _ = args, kwargs
        raise AssertionError("DatabaseConfigManager should not be constructed")

    monkeypatch.setattr(
        "sqlsaber.config.settings.ModelConfigManager",
        _fail_model_config_manager,
    )
    monkeypatch.setattr(
        "sqlsaber.database.resolver.DatabaseConfigManager",
        _fail_database_config_manager,
    )

    options = SQLSaberOptions(
        database="sqlite:///:memory:",
        settings=Config.in_memory(
            model_name="anthropic:claude-4-sonnet",
            api_keys={"anthropic": "test-key"},
        ),
    )

    saber = SQLSaber(options=options)
    try:
        assert saber.agent.agent.model.model_name == "claude-4-sonnet"
    finally:
        await saber.close()
