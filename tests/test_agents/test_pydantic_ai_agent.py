"""Tests for SQLSaberAgent overrides and lifecycle behavior."""

from types import SimpleNamespace

import pytest
from pydantic_ai.usage import UsageLimits

from sqlsaber.agents import pydantic_ai_agent as agent_module
from sqlsaber.agents.pydantic_ai_agent import SQLSaberAgent
from sqlsaber.capabilities.base import SqlSaberCapability
from sqlsaber.capabilities.plugins import PluginContext
from sqlsaber.config.settings import Config
from sqlsaber.database.sqlite import SQLiteConnection
from sqlsaber.knowledge.manager import KnowledgeManager
from sqlsaber.knowledge.sqlite_store import SQLiteKnowledgeStore
from sqlsaber.run_usage import current_usage_limits
from sqlsaber.tools.knowledge_tool import SearchKnowledgeTool
from sqlsaber_viz import capability as viz_factory


@pytest.fixture
def in_memory_db():
    """Create an in-memory SQLite connection for testing."""
    return SQLiteConnection("sqlite:///:memory:")


class TestSQLSaberAgentOverrides:
    """Test validation logic for model_name and api_key overrides."""

    def test_api_key_without_model_name_raises_error(self, in_memory_db):
        """api_key requires model_name to be specified."""
        with pytest.raises(ValueError):
            SQLSaberAgent(db_connection=in_memory_db, api_key="test-key")

    def test_model_name_and_api_key_together_accepted(self, in_memory_db):
        """Both model_name and api_key together should work."""
        agent = SQLSaberAgent(
            db_connection=in_memory_db,
            model_name="anthropic:claude-3-5-sonnet",
            api_key="test-key",
        )
        assert agent is not None
        assert agent.agent is not None
        assert agent.agent.model.model_name == "claude-3-5-sonnet"


class TestSQLSaberAgentKnowledge:
    def test_knowledge_tool_context_configured(self, in_memory_db):
        agent = SQLSaberAgent(
            db_connection=in_memory_db,
            database_name="test-db",
            model_name="anthropic:claude-3-5-sonnet",
            api_key="test-key",
        )

        tool = agent._tools.get("search_knowledge")
        assert isinstance(tool, SearchKnowledgeTool)
        assert tool.database_name == "test-db"
        assert tool.knowledge_manager is agent.knowledge_manager


class TestSQLSaberAgentDeps:
    @pytest.mark.asyncio
    async def test_run_does_not_claim_deps(self, in_memory_db, monkeypatch):
        agent = SQLSaberAgent(
            db_connection=in_memory_db,
            model_name="anthropic:claude-3-5-sonnet",
            api_key="test-key",
            capabilities=[viz_factory],
            tool_overides={
                "viz": {
                    "model_name": "openai:gpt-5-mini",
                    "api_key": "override-api-key",
                }
            },
        )

        captured: dict[str, object] = {}

        async def fake_run(prompt: str, **kwargs):
            _ = prompt
            captured.update(kwargs)
            captured["bound_usage_limits"] = current_usage_limits()
            return SimpleNamespace(output="ok")

        monkeypatch.setattr(agent.agent, "run", fake_run)

        usage_limits = UsageLimits(request_limit=200)
        await agent.run("hello", usage_limits=usage_limits)
        assert "deps" not in captured
        assert captured["usage_limits"] is usage_limits
        assert captured["bound_usage_limits"] is usage_limits

        captured.clear()
        await agent.run("use defaults")
        assert "deps" not in captured
        assert captured["usage_limits"] is None
        assert captured["bound_usage_limits"] is None
        viz = agent._tools["viz"]
        assert viz.model_overide.model_name == "openai:gpt-5-mini"
        assert viz.model_overide.api_key == "override-api-key"


class _ClosingCapability(SqlSaberCapability):
    def __init__(self) -> None:
        self.close_calls = 0
        self.contexts: list[PluginContext] = []

    def update_context(self, context: PluginContext) -> None:
        self.contexts.append(context)

    async def close(self) -> None:
        self.close_calls += 1


class TestSQLSaberAgentLifecycle:
    @pytest.mark.asyncio
    async def test_plugin_capability_survives_rebuilds_and_refreshes_context(
        self, in_memory_db
    ):
        created: list[_ClosingCapability] = []

        def factory(context: PluginContext) -> _ClosingCapability:
            capability = _ClosingCapability()
            capability.contexts.append(context)
            created.append(capability)
            return capability

        config = Config.in_memory(
            model_name="anthropic:claude-3-5-sonnet",
            api_keys={"anthropic": "test-key"},
        )
        agent = SQLSaberAgent(
            db_connection=in_memory_db, settings=config, capabilities=[factory]
        )
        plugin = created[0]

        agent.set_thinking(True)
        config.model.name = "anthropic:claude-3-5-haiku"
        agent.reload_model_settings()

        assert created == [plugin]
        assert plugin in agent.capabilities
        assert [context.main_model_name for context in plugin.contexts] == [
            "anthropic:claude-3-5-sonnet",
            "anthropic:claude-3-5-sonnet",
            "anthropic:claude-3-5-haiku",
        ]

        await agent.close()
        await agent.close()
        assert plugin.close_calls == 1

    @pytest.mark.asyncio
    async def test_failed_rebuild_preserves_plugin_and_agent_state(
        self, in_memory_db, monkeypatch
    ):
        plugin = _ClosingCapability()
        factory_calls = 0

        def factory(context: PluginContext) -> _ClosingCapability:
            nonlocal factory_calls
            factory_calls += 1
            plugin.contexts.append(context)
            return plugin

        agent = SQLSaberAgent(
            db_connection=in_memory_db,
            model_name="anthropic:claude-3-5-sonnet",
            api_key="test-key",
            capabilities=[factory],
        )
        previous_agent = agent.agent
        previous_capabilities = agent.capabilities
        previous_thinking = agent.thinking_enabled

        def fail_agent(*args, **kwargs):
            raise RuntimeError("rebuild failed")

        monkeypatch.setattr(agent_module, "Agent", fail_agent)
        with pytest.raises(RuntimeError, match="rebuild failed"):
            agent.set_thinking(not previous_thinking)

        assert factory_calls == 1
        assert agent.agent is previous_agent
        assert agent.capabilities is previous_capabilities
        assert agent.thinking_enabled is previous_thinking
        assert len(plugin.contexts) == 1

        await agent.close()
        assert plugin.close_calls == 1

    @pytest.mark.asyncio
    async def test_plugin_capabilities_are_not_shared_between_agents(self):
        created: list[_ClosingCapability] = []

        def factory(context: PluginContext) -> _ClosingCapability:
            capability = _ClosingCapability()
            capability.contexts.append(context)
            created.append(capability)
            return capability

        agents = [
            SQLSaberAgent(
                db_connection=SQLiteConnection("sqlite:///:memory:"),
                model_name="anthropic:claude-3-5-sonnet",
                api_key="test-key",
                capabilities=[factory],
            )
            for _ in range(2)
        ]

        assert len(created) == 2
        assert created[0] is not created[1]

        for agent in agents:
            await agent.close()
        assert [capability.close_calls for capability in created] == [1, 1]

    @pytest.mark.asyncio
    async def test_close_does_not_close_injected_knowledge_manager(
        self, in_memory_db, temp_dir, monkeypatch
    ):
        manager = KnowledgeManager(
            store=SQLiteKnowledgeStore(db_path=temp_dir / "knowledge.db")
        )
        agent = SQLSaberAgent(
            db_connection=in_memory_db,
            database_name="test-db",
            model_name="anthropic:claude-3-5-sonnet",
            api_key="test-key",
            knowledge_manager=manager,
        )

        close_calls = 0

        async def _track_close() -> None:
            nonlocal close_calls
            close_calls += 1

        monkeypatch.setattr(manager, "close", _track_close)
        await agent.close()

        assert close_calls == 0

    @pytest.mark.asyncio
    async def test_close_closes_sqlsaber_capabilities_once(self, in_memory_db):
        agent = SQLSaberAgent(
            db_connection=in_memory_db,
            database_name="test-db",
            model_name="anthropic:claude-3-5-sonnet",
            api_key="test-key",
        )
        capability = _ClosingCapability()
        agent.capabilities.append(capability)

        await agent.close()
        await agent.close()

        assert capability.close_calls == 1

    @pytest.mark.asyncio
    async def test_close_closes_owned_knowledge_manager_once(
        self, in_memory_db, monkeypatch
    ):
        agent = SQLSaberAgent(
            db_connection=in_memory_db,
            database_name="test-db",
            model_name="anthropic:claude-3-5-sonnet",
            api_key="test-key",
        )

        close_calls = 0

        async def _track_close() -> None:
            nonlocal close_calls
            close_calls += 1

        monkeypatch.setattr(agent.knowledge_manager, "close", _track_close)

        await agent.close()
        await agent.close()

        assert close_calls == 1
