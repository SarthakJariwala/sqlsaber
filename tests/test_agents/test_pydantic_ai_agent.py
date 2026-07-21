"""Tests for SQLSaberAgent overrides and lifecycle behavior."""

from types import SimpleNamespace

import pytest

from sqlsaber.agents.pydantic_ai_agent import SQLSaberAgent
from sqlsaber.capabilities.base import SqlSaberCapability
from sqlsaber.database.sqlite import SQLiteConnection
from sqlsaber.knowledge.manager import KnowledgeManager
from sqlsaber.knowledge.sqlite_store import SQLiteKnowledgeStore
from sqlsaber.tools.knowledge_tool import SearchKnowledgeTool


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
            return SimpleNamespace(output="ok")

        monkeypatch.setattr(agent.agent, "run", fake_run)

        await agent.run("hello")
        assert "deps" not in captured
        viz = agent._tools["viz"]
        assert viz.model_overide.model_name == "openai:gpt-5-mini"
        assert viz.model_overide.api_key == "override-api-key"


class _ClosingCapability(SqlSaberCapability):
    def __init__(self) -> None:
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1


class TestSQLSaberAgentLifecycle:
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
